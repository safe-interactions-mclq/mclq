#include <mclq/alloc.h>
#include <mclq/module.h>
#include <mclq/random.h>
#include <stdalign.h>

#ifndef max
#define max(x, y) (x) > (y) ? (x) : (y)
#endif

#ifndef min
#define min(x, y) (x) > (y) ? (y) : (x)
#endif

// clang-format off
/* initializing memory blocks. note that these are NOT threadsafe */
#define STACK_MEM_BYTES 65536

static uint8_t g_stack_mem[STACK_MEM_BYTES];
static StackAllocator g_stack = { 
    .memory = g_stack_mem, 
    .size = STACK_MEM_BYTES, 
    .offset = 0
};
static Allocator g_stack_iface = {&g_stack, stack_alloc, stack_reset};

static HeapAllocator g_heap = {
    .head = NULL
};
static Allocator g_heap_iface = {&g_heap, heap_alloc, heap_reset};

static CompositeAllocator g_comp_ctx = {
    .primary = &g_stack_iface,
    .secondary = &g_heap_iface
};

#ifndef DEBUG
static Allocator g_comp_iface = {&g_comp_ctx, comp_alloc, comp_reset};
#else 
static Allocator _g_comp_iface = {&g_comp_ctx, comp_alloc, comp_reset};
static DebugAllocator g_debug_ctx = {
    .delegate = &_g_comp_iface, 
    .log_stream = NULL,         
    .name = "MCLQ_Main"
};

static Allocator g_comp_iface = {&g_debug_ctx, debug_alloc, debug_reset};

#endif
static void mclq_cleanup(void) {
	g_comp_iface.reset(g_comp_iface.context);
}

/* end initialize memory blocks */
// clang-format on

// we assume that x is already allocated
static inline void rand_double_arr(arr *x, double lb, double ub) {
	uint32_t n = x->dim * x->N;
	for (uint32_t i = 0; i < n; i++) {
		x->data[i] = randdouble(lb, ub);
		x->data[i] = min(1.0, max(-1.0, x->data[i]));
	}
}

static inline void rand_noise_to_arr(arr *x, double lb, double ub) {
	uint32_t n = x->dim * x->N;
	for (uint32_t i = 0; i < n; i++) {
		x->data[i] += randdouble(lb, ub);
		x->data[i] = min(1.0, max(-1.0, x->data[i]));
	}
}

static inline double simulate_rollout(
	dyn_func_t dynamics, cost_func_t objective, arr *xN, arr *uN, arr *wN) {
	assert(uN->N == wN->N);

	const uint32_t horizon = uN->N;
	const uint32_t xdim = xN->dim;
	const uint32_t udim = uN->dim;
	const uint32_t wdim = wN->dim;

	double cost = 0.0;
	double *x = xN->data;
	double *x_next = xN->data + xN->dim;
	double *u = uN->data;
	double *w = wN->data;

	for (uint32_t i = 0; i < horizon; i++) {
		cost += objective(x, u, w);
		dynamics(x_next, x, u, w);
		x_next += xdim;
		x += xdim;
		u += udim;
		w += wdim;
	}

	return cost;
}

static inline double dnorm(const arr *x, const arr *y) {
	int m = y->N * y->dim;
	assert(x->N == y->N);
	assert(x->dim == y->dim);
	double n = 0.0;
	double tmp = 0.0;
	for (int i = 0; i < m; i++) {
		tmp = x->data[i] - y->data[i];
		n += tmp * tmp;
	}
	return sqrt(n);
}

PyObject *mclq_dmh(PyObject *self, PyObject *args) {
	// python parameters
	Py_buffer state_buf, u_seed_buf, w_seed_buf;
	PyObject *result;
	// C parameters
	unsigned long long dyn_addr, cost_addr, human_addr;
	int is_deterministic, iter_out, iter_in, dx, du, dw, horizon;
	double beta, lambda, epsilon;

	double tmp, cost, cost_i, cost_j;
	dyn_func_t dyn_func;
	cost_func_t cost_func;
	human_func_t human_policy;

	tmp = 0.0;
	cost = 0.0;
	cost_i = 0.0;
	cost_j = 0.0;

	if (!PyArg_ParseTuple(args,
						  "y*y*y*KKKiiiiiiiddd",
						  &state_buf,
						  &u_seed_buf,
						  &w_seed_buf,
						  &dyn_addr,
						  &cost_addr,
						  &human_addr,
						  &is_deterministic,
						  &iter_out,
						  &iter_in,
						  &dx,
						  &du,
						  &dw,
						  &horizon,
						  &beta,
						  &lambda,
						  &epsilon)) {
		return NULL;
	}
#ifdef DEBUG
	printf("Iter out: %d\n", iter_out);
	printf("Iter in : %d\n", iter_in);
	printf("dx      : %d\n", dx);
	printf("du      : %d\n", du);
	printf("dw      : %d\n", dw);
	printf("horizon : %d\n", horizon);
	printf("beta    : %lf\n", beta);
	printf("lambda   : %lf\n", lambda);
	printf("epsilon : %lf\n", epsilon);
#endif
	// ensure that memory is not fragmented
	mclq_cleanup();

	if (is_deterministic) {
		iter_in = 1;
	}

	arr x = {NULL, dx, horizon + 1};
	arr xCand = {NULL, dx, horizon + 1};
	arr u = {NULL, du, horizon};
	arr w = {NULL, dw, horizon};
	arr uN = {NULL, du, horizon};
	arr wN = {NULL, dw, horizon};
	arr uCand = {NULL, du, horizon};
	arr wCand = {NULL, dw, horizon};

	dyn_func = (dyn_func_t)(dyn_addr);
	cost_func = (cost_func_t)(cost_addr);
	human_policy = (human_func_t)(human_addr);

	x.data = (double *)g_comp_iface.alloc(g_comp_iface.context,
										  sizeof(double) * dx * (horizon + 1),
										  alignof(double));

	xCand.data =
		(double *)g_comp_iface.alloc(g_comp_iface.context,
									 sizeof(double) * dx * (horizon + 1),
									 alignof(double));

	u.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * du * horizon, alignof(double));

	w.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * dw * horizon, alignof(double));

	uN.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * du * horizon, alignof(double));

	wN.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * dw * horizon, alignof(double));

	uCand.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * du * horizon, alignof(double));

	wCand.data = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * dw * horizon, alignof(double));

	// initialize values
	memcpy(x.data, state_buf.buf, dx * sizeof(double));
	memcpy(u.data, u_seed_buf.buf, du * horizon * sizeof(double));
	memcpy(w.data, w_seed_buf.buf, dw * horizon * sizeof(double));

	const size_t u_size = uN.N * uN.dim * sizeof(double);
	const size_t w_size = wN.N * wN.dim * sizeof(double);
	const size_t x_size = x.N * x.dim * sizeof(double);

	memcpy(xCand.data, x.data, x_size);
	memcpy(uN.data, u.data, u_size);
	memcpy(wN.data, w.data, w_size);

	double *tmp_ptr;

	double *current_state = (double *)g_comp_iface.alloc(
		g_comp_iface.context, sizeof(double) * dx, alignof(double));
	for (int i = 0; i < dx; i++) {
		current_state[i] = xCand.data[i];
	}

	// Reset cost baseline
	const int USE_NBALL_NOISE_APPROXIMATION = 0;
	Py_BEGIN_ALLOW_THREADS;
	if (human_policy && lambda <= 0.0) {
		cost = simulate_rollout(dyn_func, cost_func, &x, &u, &w);
		for (int i = 0; i < iter_out; i++) {
			for (int j = 0; j < iter_in; j++) {
				memcpy(wCand.data, w.data, w_size);
				cost_j = 0.0;
				for (int k = 0; k < horizon; k++) {
					// use x instead of xCand because we know that it is
					// the state trajectory that is most likely to occur
					// (it uses the best U found so far)
					// we also need to step the environment state according
					// to this new x, not the old x. since the entire process
					// is deterministic, we can also calculate the cost
					// for rollout here too, and make a second call to
					// `simulate_rollout` unnecessary.
					human_policy(x.data + k * dx, wCand.data + k * dw);
					cost_j += cost_func(
						x.data + k * dx, u.data + k * du, wCand.data + k * dw);
					dyn_func(x.data + (k + 1) * dx,
							 x.data + k * dx,
							 u.data + k * du,
							 wCand.data + k * dw);
				}
				tmp = beta * (cost_j - cost);
				if (cost_j > cost ||
					(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
					cost = cost_j;
					memcpy(w.data, wCand.data, w_size);
				}
			}
			memcpy(uCand.data, u.data, u_size);
			rand_noise_to_arr(&uCand, -epsilon, epsilon);
			cost_i = simulate_rollout(dyn_func, cost_func, &xCand, &uCand, &w);
			tmp = beta * (cost - cost_i);
			if (cost_i < cost ||
				(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
				cost = cost_i;
				memcpy(u.data, uCand.data, u_size);
			}
		}
	} else if (human_policy && USE_NBALL_NOISE_APPROXIMATION) {
		const double max_noise = sqrt(lambda / (dw * horizon));
		// then there is a safety margin that we should consider, but still
		// refer to the human policy
		cost = simulate_rollout(dyn_func, cost_func, &x, &u, &w);
		for (int i = 0; i < iter_out; i++) {
			for (int j = 0; j < iter_in; j++) {
				memcpy(wCand.data, w.data, w_size);
				for (int k = 0; k < horizon; k++) {
					human_policy(x.data + k * dx, wCand.data + k * dw);
					dyn_func(x.data + (k + 1) * dx,
							 x.data + k * dx,
							 u.data + k * du,
							 wCand.data + k * dw);
				}
				// this is a very rough approximation of the original equation
				// presented in MCLQ. It explores an n-ball centered at wCand,
				// so it does not fully explore the state space encompassed by
				// ||noise||^2 <= lambda. the approximation gets worse as the
				// size of u increases
				rand_noise_to_arr(&wCand, -max_noise, max_noise);
				cost_j =
					simulate_rollout(dyn_func, cost_func, &xCand, &u, &wCand);
				tmp = beta * (cost_j - cost);
				if (cost_j > cost ||
					(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
					cost = cost_j;
					memcpy(w.data, wCand.data, w_size);
				}
			}
			memcpy(uCand.data, u.data, u_size);
			rand_noise_to_arr(&uCand, -epsilon, epsilon);
			cost_i = simulate_rollout(dyn_func, cost_func, &xCand, &uCand, &w);
			tmp = beta * (cost - cost_i);
			if (cost_i < cost ||
				(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
				cost = cost_i;
				memcpy(u.data, uCand.data, u_size);
			}
		}
	} else if (human_policy) {
		// then there is a safety margin that we should consider, but still
		// refer to the human policy
		// we need another temporary variable here to store the predicted human
		// action
		arr wHuman = {NULL, wCand.dim, wCand.N};
		wHuman.data =
			(double *)g_comp_iface.alloc(g_comp_iface.context,
										 sizeof(double) * dw * horizon,
										 alignof(double));
		const double noise_numer = sqrt(lambda);
		// begin mclq
		cost = simulate_rollout(dyn_func, cost_func, &x, &u, &w);
		for (int i = 0; i < iter_out; i++) {
			for (int j = 0; j < iter_in; j++) {
				memcpy(wHuman.data, w.data, w_size);
				// verify that the distance is within lambda of human's model
				// action. if it is not, step until it is
				for (int k = 0; k < horizon; k++) {
					human_policy(x.data + k * dx, wHuman.data + k * dw);
					dyn_func(x.data + (k + 1) * dx,
							 x.data + k * dx,
							 u.data + k * du,
							 wHuman.data + k * dw);
				}
				memcpy(wHuman.data, w.data, w_size);
				rand_noise_to_arr(&wCand, -epsilon, epsilon);
				// now that we have a candidate action, we check if it is close
				// enough to the human's action (unlikely) otherwise, we use z =
				// (1 - sqrt(c) / ||x - y||) (x - y)
				double noise_denom;
				if ((noise_denom = dnorm(&wCand, &wHuman)) > lambda) {
					const double mult = 1 - noise_numer / noise_denom;
					const double multm1 = 1 - mult;
					for (int k = 0; k < dw * horizon; k++) {
						wCand.data[k] =
							mult * wHuman.data[k] + multm1 * wCand.data[k];
					}
				}
				cost_j =
					simulate_rollout(dyn_func, cost_func, &xCand, &u, &wCand);
				tmp = beta * (cost_j - cost);
				if (cost_j > cost ||
					(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
					cost = cost_j;
					memcpy(w.data, wCand.data, w_size);
				}
			}
			memcpy(uCand.data, u.data, u_size);
			rand_noise_to_arr(&uCand, -epsilon, epsilon);
			cost_i = simulate_rollout(dyn_func, cost_func, &xCand, &uCand, &w);
			tmp = beta * (cost - cost_i);
			if (cost_i < cost ||
				(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
				cost = cost_i;
				memcpy(u.data, uCand.data, u_size);
			}
		}
	} else {
		cost = simulate_rollout(dyn_func, cost_func, &x, &u, &w);
		for (int i = 0; i < iter_out; i++) {
			for (int j = 0; j < iter_in; j++) {
				memcpy(wCand.data, w.data, w_size);
				rand_noise_to_arr(&wCand, -epsilon, epsilon);
				cost_j =
					simulate_rollout(dyn_func, cost_func, &xCand, &u, &wCand);
				tmp = beta * (cost_j - cost);
				if (cost_j > cost ||
					(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
					cost = cost_j;
					memcpy(w.data, wCand.data, w_size);
				}
			}
			memcpy(uCand.data, u.data, u_size);
			rand_noise_to_arr(&uCand, -epsilon, epsilon);
			cost_i = simulate_rollout(dyn_func, cost_func, &xCand, &uCand, &w);
			tmp = beta * (cost - cost_i);
			if (cost_i < cost ||
				(tmp > -50 && tmp > log(randdouble(1e-10, 1.0)))) {
				cost = cost_i;
				memcpy(u.data, uCand.data, u_size);
			}
		}
	}
	Py_END_ALLOW_THREADS;

	result = PyList_New(horizon * (du + dw));
	for (int i = 0; i < (du * horizon); i++) {
		PyList_SetItem(result, i, PyFloat_FromDouble(u.data[i]));
	}
	for (int i = 0; i < (dw * horizon); i++) {
		PyList_SetItem(result, du * horizon + i, PyFloat_FromDouble(w.data[i]));
	}
	PyBuffer_Release(&state_buf);
	PyBuffer_Release(&u_seed_buf);
	PyBuffer_Release(&w_seed_buf);
	return result;
}

// clang-format off

static PyMethodDef MyMethods[] = {
    {"mclq_dmh", mclq_dmh, METH_VARARGS, "Run Double MH Sampling in C"}, 
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef MyModule = {
    PyModuleDef_HEAD_INIT, 
    "mclq", 
    NULL, 
    -1, 
    MyMethods
};

PyMODINIT_FUNC PyInit_libmclq(void) { 
	srand(time(0));
	#ifdef DEBUG
	g_debug_ctx.log_stream = stderr;
	#endif
    if (Py_AtExit(mclq_cleanup) != 0) {
        fprintf(stderr, "mclq: Failed to register cleanup function.\n");
    }
    return PyModule_Create(&MyModule); 
}

// clang-format on
