import numpy as np
from mclq import MCLQProblem, MCLQController


class HighwayProblem(MCLQProblem):
    def __init__(self, lane_width=3.5, target_speed=20.0, dt=0.1):
        super().__init__(xdim=4, udim=2, wdim=2)
        self.lane_width = lane_width
        self.target_speed = target_speed
        self.dt = dt

    def dynamics(self, x_next, x, u, w):
        x_next[0] = x[0] + x[2] * np.cos(x[3]) * self.dt
        x_next[1] = x[1] + x[2] * np.sin(x[3]) * self.dt
        x_next[2] = x[2] + u[0] * self.dt
        x_next[3] = x[3] + u[1] * self.dt

    def cost_fn(self, x, u, w):
        lane_penalty = 0.0
        if abs(x[1]) > self.lane_width / 2:
            lane_penalty = 100.0 * (abs(x[1]) - self.lane_width / 2) ** 2

        speed_error = (x[2] - self.target_speed) ** 2
        return speed_error + lane_penalty


def test_highway_constraints():
    print("\n--- Testing Nonlinear Highway Constraints ---")
    prob = HighwayProblem()
    ctrl = MCLQController(prob, horizon=40)
    state = np.array([0.0, 2.0, 15.0, 0.0])
    u, w = ctrl.solve(state)
    print(f"Computed Accel: {u[0]:.2f}, Steer: {u[1]:.2f}")
    assert u.shape == (2,)
    print("Highway test passed.")
