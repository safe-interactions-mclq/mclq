import numpy as np
import torch


class Env:
    def __init__(
        self,
        xlb: float | np.ndarray = -10.0,
        xub: float | np.ndarray = 10.0,
        glb: float | None = None,
        gub: float | None = None,
    ):
        self.xdim = 4
        self.udim = 2
        self.wdim = 2
        self.gdim = 2
        self.n_agents = 2
        self.x_lb = xlb
        self.x_ub = xub
        if glb is None:
            if isinstance(xlb, float):
                glb = xlb
            else:
                glb = xlb[0:2]  # type: ignore
        if gub is None:
            if isinstance(xub, float):
                gub = xub
            else:
                gub = xub[0:2]  # type: ignore
        self.glb = glb
        self.gub = gub
        self.reset()
        return

    def reset(self):
        x = np.random.uniform(self.x_lb, self.x_ub, (self.xdim * self.n_agents, 1))
        x[2] /= 50.0
        x[6] /= 50.0
        self.u_goal = np.random.uniform(self.glb, self.gub, (self.gdim, 1))  # type: ignore
        self.w_goal = np.random.uniform(self.glb, self.gub, (self.gdim, 1))  # type: ignore
        self.x = np.concatenate((x, self.u_goal, self.w_goal), axis=0)
        return

    def dynamics(self, x, u, w):
        L_R = 2.0
        L_F = 1.5
        if isinstance(x, np.ndarray):
            x_dot = np.zeros((len(x), 1))
            cos = np.cos
            sin = np.sin
            atan = np.atan
            tan = np.tan
        else:
            x_dot = torch.zeros((len(x), 1))
            cos = torch.cos
            sin = torch.sin
            atan = torch.atan
            tan = torch.tan

        _u = u[0]
        _w = w[0]

        v1 = x[2]
        t1 = x[3]
        v2 = x[6]
        t2 = x[7]

        beta1 = atan(tan(u[1]) * (L_R / (L_R + L_F)))
        beta2 = atan(tan(w[1]) * (L_R / (L_R + L_F)))

        vd1 = _u
        xd1 = v1 * cos(t1 + beta1)
        yd1 = v1 * sin(t1 + beta1)
        td1 = v1 / L_R * sin(beta1)  # type: ignore

        vd2 = _w
        xd2 = v2 * cos(t2 + beta2)
        yd2 = v2 * sin(t2 + beta2)
        td2 = v2 / L_R * sin(beta2)  # type: ignore

        x_dot[0] = 0.5 * xd1
        x_dot[1] = 0.5 * yd1
        x_dot[2] = 0.2 * vd1
        x_dot[3] = td1
        x_dot[4] = 0.5 * xd2
        x_dot[5] = 0.5 * yd2
        x_dot[6] = 0.2 * vd2
        x_dot[7] = td2

        return x + x_dot

    def robot_cost(
        self,
        x: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        w: np.ndarray | torch.Tensor,
    ):
        if isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            assert isinstance(w, torch.Tensor)
            norm = torch.linalg.norm
            exp = torch.exp
            e = torch.e
            c = torch.FloatTensor([0.0]).requires_grad_(True)
        else:
            norm = np.linalg.norm
            exp = np.exp
            e = np.e
            c = 0.0
        z = -5.0 / e * norm(x[0:2] - x[4:6])
        c = c + 10.0 * exp(z)  # type: ignore
        c = c + 100.0 * norm(x[0:2] - x[8:10])
        c = c + 2.0 * norm(u)
        c = c + 10.0 * norm(x[2])
        return c

    def human_cost(
        self,
        x: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        w: np.ndarray | torch.Tensor,
    ):
        if isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            assert isinstance(w, torch.Tensor)
            norm = torch.linalg.norm
            c = torch.FloatTensor([0.0]).requires_grad_(True)
        else:
            norm = np.linalg.norm
            c = 0.0
        xhat = self.dynamics(x, u, w)
        c = c + 100.0 * norm(xhat[4:6] - xhat[10:12])
        c = c + 2.0 * norm(w)
        c = c + 10.0 * norm(xhat[6])
        return c

    def step(self, u: np.ndarray, w: np.ndarray):
        self.x = self.dynamics(self.x, u, w)
