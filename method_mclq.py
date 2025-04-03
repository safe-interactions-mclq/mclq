import numpy as np
from method_npg import NPGSolver


class MCLQSolver:
    def __init__(
        self,
        env,
        Tout: int = 200,
        Tin: int = 50,
        seed_npg: bool = False,
        horizon: int = 10,
        lmbda: float = 0.1,
        beta: float = 10.0,
    ):
        self.env = env
        self.dynamics = env.dynamics
        self.cost = env.robot_cost
        self.Tout = Tout
        self.Tin = Tin
        self.do_seed = seed_npg
        self.seeded = False
        self.T = horizon
        self.npg = NPGSolver(env, Tout=Tout, Tin=Tin)
        self.us = np.random.uniform(-1.0, 1.0, (self.T, self.env.udim, 1))
        self.ws = np.random.uniform(-1.0, 1.0, (self.T, self.env.wdim, 1))
        self.lmbda = lmbda
        self.beta = beta

    def traj_dynamics(self, x: np.ndarray, us: np.ndarray, ws: np.ndarray):
        xn = x.copy()
        for t in range(self.T):
            xn = self.dynamics(xn, us[t], ws[t])
        return xn

    def traj_cost(self, x: np.ndarray, us: np.ndarray, ws: np.ndarray):
        xn = x.copy()
        c = 0.0
        for t in range(self.T):
            c += self.cost(xn, us[t], ws[t])
            xn = self.dynamics(xn, us[t], ws[t])
        return c

    def solve(self, x):
        if (self.us is None or self.ws is None) and not self.seeded and self.do_seed:
            self.us, self.ws = self.npg.solve(x)
            self.seeded = True

        cost = np.inf
        delta_us = np.zeros(self.us.shape)
        delta_ws = np.zeros(self.ws.shape)
        for _ in range(self.Tout):
            for _ in range(self.Tin):
                dw = np.random.uniform(-self.lmbda, self.lmbda, self.ws.shape)
                cost_j = self.traj_cost(x, self.us + delta_us, self.ws + delta_ws + dw)
                if np.exp(self.beta * (cost_j - cost)) > np.random.rand():
                    cost = cost_j
                    delta_ws = dw
            du = np.random.uniform(self.lmbda, self.lmbda, self.us.shape)
            cost_i = self.traj_cost(x, self.us + delta_us + du, self.ws + delta_ws)
            if np.exp(self.beta * (cost - cost_i)) > np.random.rand():
                cost = cost_i
                delta_us = du
        return self.us + delta_us, self.ws + delta_ws
