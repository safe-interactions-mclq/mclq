import numpy as np
from abc import ABC, abstractmethod
from numba import njit, cfunc, types, carray
from collections import namedtuple

try:
    from warnings import deprecated
except ImportError:
    try:
        from typing_extensions import deprecated
    except ImportError:

        def deprecated(message, *args, **kwargs):
            def decorator(obj):
                return obj

            return decorator


from .utils import _force_jit_func


class MCLQProblem(ABC):
    """inherit from this to define your specific problem."""

    def __init__(
        self, xdim: int, udim: int, wdim: int, deterministic_human: bool = False
    ):
        self.dx = xdim
        self.du = udim
        self.dw = wdim
        self.deterministic_human = deterministic_human

    @abstractmethod
    def dynamics(self, x_next: carray, x: carray, u: carray, w: carray) -> None:
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, x: carray, u: carray, w: carray) -> float:
        raise NotImplementedError

    # @abstractoptional
    def human_policy(self, x: carray, w: carray) -> None:
        """
        Optional: Returns the human's response w to state x.
        If not implemented, default DMH sampling is used for the human's action as well.
        """
        raise NotImplementedError


class MCLQController:
    def __init__(
        self,
        problem: MCLQProblem,
        horizon: int,
        beta=10.0,
        epsilon=1.0,
        safety_margin=0.0,
    ):
        self.problem = problem
        self.horizon = horizon
        self.beta = beta
        self.epsilon = epsilon
        self.safety_margin = safety_margin

        self._dyn_cfunc = self._generate_dyn_wrapper()
        self._cost_cfunc = self._generate_cost_wrapper()
        self._human_cfunc = self._generate_human_wrapper()
        if self._human_cfunc is None:
            self._human_cfunc_address = 0
        else:
            self._human_cfunc_address = self._human_cfunc.address
        self.u_seed = np.zeros(problem.du * horizon, dtype=np.float64, order="C")
        self.w_seed = np.zeros(problem.dw * horizon, dtype=np.float64, order="C")

    def _get_problem_params(self):
        """Extracts all numeric attributes from the user's problem instance."""
        attrs = {}
        for k, v in vars(self.problem).items():
            if isinstance(v, (int, float, np.ndarray, np.float64, np.int64)) or hasattr(
                v, "__dict__"
            ):
                attrs[k] = v
        ParamStruct = namedtuple("ParamStruct", attrs.keys())
        return ParamStruct(**attrs)

    def _generate_dyn_wrapper(self):
        dx, du, dw = self.problem.dx, self.problem.du, self.problem.dw
        params = self._get_problem_params()
        user_logic_jitted = _force_jit_func(self.problem.dynamics.__func__)

        @njit
        def captured_logic(xn, x, u, w, p):
            user_logic_jitted(p, xn, x, u, w)

        @cfunc(
            types.void(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.CPointer(types.float64),
            )
        )
        def _dyn(xn_p, x_p, u_p, w_p):
            captured_logic(
                carray(xn_p, (dx,)),
                carray(x_p, (dx,)),
                carray(u_p, (du,)),
                carray(w_p, (dw,)),
                params,
            )

        return _dyn

    def _generate_cost_wrapper(self):
        dx, du, dw = self.problem.dx, self.problem.du, self.problem.dw
        params = self._get_problem_params()
        user_logic_jitted = _force_jit_func(self.problem.cost_fn.__func__)

        @njit
        def captured_logic(x, u, w, p):
            return user_logic_jitted(p, x, u, w)

        @cfunc(
            types.float64(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
                types.CPointer(types.float64),
            )
        )
        def _cost(x_p, u_p, w_p):
            return captured_logic(
                carray(x_p, (dx,)), carray(u_p, (du,)), carray(w_p, (dw,)), params
            )

        return _cost

    def _generate_human_wrapper(self):
        is_not_overriden = (
            self.problem.human_policy.__func__ is MCLQProblem.human_policy
        )

        if is_not_overriden:
            return None
        dx, dw = self.problem.dx, self.problem.dw
        params = self._get_problem_params()
        user_logic_jitted = _force_jit_func(self.problem.human_policy.__func__)

        @njit
        def captured_logic(x, w, p):
            return user_logic_jitted(p, x, w)

        @cfunc(
            types.void(
                types.CPointer(types.float64),
                types.CPointer(types.float64),
            )
        )
        def _human(x_p, w_p):
            return captured_logic(carray(x_p, (dx,)), carray(w_p, (dw,)), params)

        return _human

    def solve(self, state: np.ndarray, tout=200, tin=100):
        from . import libmclq

        if state.size != self.problem.dx:
            raise ValueError(f"State size {state.size} != dx {self.problem.dx}")

        state = np.ascontiguousarray(state, dtype=np.float64)

        u_seed_contig = np.ascontiguousarray(self.u_seed, dtype=np.float64)
        w_seed_contig = np.ascontiguousarray(self.w_seed, dtype=np.float64)

        res_ptr = libmclq.mclq_dmh(
            state,
            u_seed_contig,
            w_seed_contig,
            self._dyn_cfunc.address,
            self._cost_cfunc.address,
            self._human_cfunc_address,
            int(self.problem.deterministic_human),
            int(tout),
            int(tin),
            int(self.problem.dx),
            int(self.problem.du),
            int(self.problem.dw),
            int(self.horizon),
            float(self.beta),
            float(self.safety_margin),
            float(self.epsilon),
        )
        total_size = self.horizon * (self.problem.du + self.problem.dw)
        res_array = np.ctypeslib.as_array(res_ptr, shape=(total_size,))
        split = self.horizon * self.problem.du
        u_traj = res_array[:split].reshape(self.horizon, self.problem.du)
        w_traj = res_array[split:].reshape(self.horizon, self.problem.dw)

        self.u_seed[:] = np.vstack([u_traj[1:], u_traj[-1]]).flatten()
        self.w_seed[:] = np.vstack([w_traj[1:], w_traj[-1]]).flatten()

        return u_traj[0], w_traj[0]
