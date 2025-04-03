import numpy as np
from mclq import MCLQProblem, MCLQController


class QuadrotorProblem(MCLQProblem):
    def __init__(self, mass=1.2, damping=0.1, dt=0.05):
        # dx=6, du=3 (forces), dw=3 (wind/noise)
        super().__init__(xdim=6, udim=3, wdim=3)
        self.mass = mass
        self.damping = damping
        self.dt = dt
        self.gravity = 9.81

    def dynamics(self, x_next, x, u, w):
        for i in range(3):
            x_next[i] = x[i] + x[i + 3] * self.dt
            accel = (u[i] - self.damping * x[i + 3] + w[i]) / self.mass
            if i == 2:
                accel -= self.gravity
            x_next[i + 3] = x[i + 3] + accel * self.dt

    def cost_fn(self, x, u, w):
        pos_error = x[0] ** 2 + x[1] ** 2 + x[2] ** 2
        effort = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        return pos_error + 0.1 * effort


def test_high_dim_quadrotor():
    print("\n--- Testing High-Dimensional Quadrotor ---")
    prob = QuadrotorProblem(mass=1.5, damping=0.2)
    ctrl = MCLQController(prob, horizon=30)
    state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    u, w = ctrl.solve(state)
    print(f"Action (U) computed: {u}")
    assert u.shape == (3,)
    print("High-dim test passed.")
