import numpy as np
import time
from mclq import MCLQProblem, MCLQController

DX, DU, DW = 4, 2, 2
DT = 0.1


class MyRobotProblem(MCLQProblem):
    def __init__(self, dt=0.1):
        super().__init__(xdim=DX, udim=DU, wdim=DW)
        self.dt = dt

    def dynamics(self, x_next, x, u, w):
        for i in range(2):
            x_next[i] = x[i] + u[i] * self.dt
            x_next[i + 2] = x[i + 2] + w[i] * self.dt

    def cost_fn(self, x, u, w):
        dist_sq = (x[0] - x[2]) ** 2 + (x[1] - x[3]) ** 2
        return x[0] ** 2 + x[1] ** 2 - 5.0 * min(dist_sq, 1.0)


def test_mclq_simulation():
    problem = MyRobotProblem(dt=DT)
    controller = MCLQController(problem, horizon=50, beta=10.0, epsilon=1.0)
    state = np.array([4.0, 4.0, 0.0, 0.0], dtype=np.float64)
    for i in range(10):
        t0 = time.perf_counter()
        u_act, w_act = controller.solve(state)
        state[0] += u_act[0] * DT
        state[1] += u_act[1] * DT
        state[2] += w_act[0] * DT
        state[3] += w_act[1] * DT
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(
            f"Iter {i:<3} | Time: {elapsed_ms:5.2f}ms | Pos: {state[0]:6.2f}, {state[1]:6.2f}"
        )
    assert not np.allclose(state[:2], [4.0, 4.0]), "Robot failed to move!"
    print("Test complete. The API is working correctly.")

if __name__ == "__main__":
    test_mclq_simulation()