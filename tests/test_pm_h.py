import numpy as np
from mclq import MCLQProblem, MCLQController


class PointMassHuman(MCLQProblem):
    def __init__(self):
        super().__init__(xdim=4, udim=2, wdim=2)
        self.dt = 0.1

    def dynamics(self, x_next, x, u, w):
        x_next[0] = x[0] + u[0] * self.dt
        x_next[1] = x[1] + u[1] * self.dt
        x_next[2] = x[2] + w[0] * self.dt
        x_next[3] = x[3] + w[1] * self.dt

    def cost_fn(self, x, u, w):
        dist_to_goal = (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2
        dist_to_hum = (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2
        dist_to_hum = max(dist_to_hum, 2.0)
        control_cost = u[0] ** 2 + u[1] ** 2
        convergence = w[0] ** 2 + w[1] ** 2

        return dist_to_goal - dist_to_hum + 0.1 * control_cost - 0.5 * convergence

    def human_policy(self, x, w):
        # move straight to the origin with 0.5 * unit magnitude
        norm = (x[2] ** 2 + x[3] ** 2) ** 0.5 + 1.0
        w[0] = -x[2] / norm
        w[1] = -x[3] / norm


def test_pm_h():
    print("\n--- Testing Point Mass Env w/ Human ---")
    prob = PointMassHuman()
    ctrl = MCLQController(prob, horizon=40)
    state = np.array([10.0, 10.0, 7.5, 4.0])
    u, w = ctrl.solve(state)
    prob.dynamics(state, state, u, w)
    print(state, u, w)
    print("\n--- Test Passed ---")
    
if __name__ == "__main__":
    test_pm_h()