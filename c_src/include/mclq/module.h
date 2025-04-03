#ifndef MCLQ_MODULE_H
#define MCLQ_MODULE_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
	double *data;
	const uint32_t dim;
	const uint32_t N;
} arr;

// Function pointer types for Python-passed JIT functions
typedef double (*cost_func_t)(double *x, double *u, double *w);
typedef void (*dyn_func_t)(double *x_next, double *x, double *u, double *w);
typedef void (*human_func_t)(double *x, double *w);

#endif
