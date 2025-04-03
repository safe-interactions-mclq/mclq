import numpy as np
from mclq import MCLQProblem, MCLQController


class SpringMassProblem(MCLQProblem):
    def __init__(self, k=10.0, b=0.5, dt=0.01):
        super().__init__(xdim=2, udim=1, wdim=1)
        self.k = k
        self.b = b
        self.dt = dt

    def dynamics(self, x_next, x, u, w):
        x_next[0] = x[0] + x[1] * self.dt
        accel = -self.k * x[0] - self.b * x[1] + u[0] + w[0]
        x_next[1] = x[1] + accel * self.dt

    def cost_fn(self, x, u, w):
        return 10.0 * x[0] ** 2 + 1.0 * x[1] ** 2 + 0.01 * u[0] ** 2


def test_spring_mass():
    print("\n--- Testing Spring-Mass with Long Horizon ---")
    prob = SpringMassProblem(k=25.0)
    ctrl = MCLQController(prob, horizon=100)

    state = np.array([1.0, 0.0])
    u, w = ctrl.solve(state, tout=500, tin=200)

    print(f"Control force: {u[0]:.4f}")
    assert isinstance(u[0], float)
    print("Spring-mass test passed.")
