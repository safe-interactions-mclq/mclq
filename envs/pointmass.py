import numpy as np
import torch


class Env:
    def __init__(self, xlb: float | np.ndarray = -10.0, xub: float | np.ndarray = 10.0):
        self.xdim = 2
        self.udim = 2
        self.wdim = 2
        self.gdim = 2
        self.n_agents = 2
        self.x_lb = xlb
        self.x_ub = xub
        self.reset()
        return

    def reset(self):
        x = np.random.uniform(self.x_lb, self.x_ub, (self.xdim * self.n_agents, 1))
        self.u_goal = np.random.uniform(self.x_lb, self.x_ub, (self.xdim, 1))
        self.w_goal = np.random.uniform(self.x_lb, self.x_ub, (self.xdim, 1))
        self.x = np.concatenate((x, self.u_goal, self.w_goal), axis=0)
        return

    def robot_cost(self, x, u, w):
        if isinstance(x, np.ndarray):
            norm = np.linalg.norm
            c = 0.0
        else:
            norm = torch.linalg.norm
            c = torch.FloatTensor([0.0]).requires_grad_(True)
        c += 10.0 * norm(x[0:2] - x[4:6])  # minimize distance to goal
        c -= 2.0 * norm(x[0:2] - x[2:4])  # maximize distance from agent
        c += 0.5 * norm(u)  # penalize action size
        return c

    def human_cost(self, x, u, w):
        if isinstance(x, np.ndarray):
            norm = np.linalg.norm
            c = 0.0
        else:
            norm = torch.linalg.norm
            c = torch.FloatTensor([0.0]).requires_grad_(True)
        c += 10.0 * norm(x[2:4] - x[6:8])
        c += 0.5 * norm(w)
        return c

    def dynamics(self, x, u, w):
        if isinstance(x, np.ndarray):
            xn = x.copy()
        else:
            xn = x.clone()
        xn[0:2] += u
        xn[2:4] += w
        return xn

    def step(self, u: np.ndarray, w: np.ndarray):
        self.x = self.dynamics(self.x, u, w)
