import numpy as np
import scipy as sp

from typing import List

from utils.utils import linearize, quadraticize


class NPGSolver:
    def __init__(
        self,
        env,
        Tout: int = 500,
        Tin: int = 200,
        tau1: float = 1e-3,
        tau2: float = 1e-3,
        break_on_eig: bool = False,
        horizon: int = 10,
    ):
        self.dynamics = env.dynamics
        self.cost = env.robot_cost
        self.Tout = Tout
        self.Tin = Tin
        self.tau1 = tau1
        self.tau2 = tau2
        self.do_break = break_on_eig
        self.horizon = horizon
        self.env = env

    def feedback(self, x, K, L):
        u = -K @ x
        w = -L @ x
        return self.dynamics(x, u, w), u, w

    def solve(self, x):

        m = len(x)
        n = self.env.udim
        d = self.env.wdim
        N = self.horizon
        xs = [x.copy()]
        Qs = []
        Rus = []
        Rws = []
        Abar = np.zeros((m * (N + 1), m * (N + 1)))
        Bbar = np.zeros((m * (N + 1), d * N))
        Dbar = np.zeros((m * (N + 1), n * N))
        for t in range(N):
            u = np.random.uniform(-1.0, 1.0, (n, 1))
            w = np.random.uniform(-1.0, 1.0, (d, 1))
            A, B, D = linearize(self.dynamics, xs[t], u, w)
            _, _, Q, _, Ru, _, Rw = quadraticize(
                self.cost,
                xs[t],
                u,
                w,
            )  # initial
            Rw = 100.0 * np.eye(d)
            Abar[m * (t + 1) : m * (t + 2), m * t : m * (t + 1)] = A
            Bbar[m * (t + 1) : m * (t + 2), d * (t) : d * (t + 1)] = B
            Dbar[m * (t + 1) : m * (t + 2), n * (t) : n * (t + 1)] = D
            Qs.append(Q)
            Rus.append(Ru)
            Rws.append(Rw)
            xs.append(self.dynamics(xs[t], u, w))
        u = np.random.uniform(-1.0, 1.0, (n, 1))
        w = np.random.uniform(-1.0, 1.0, (d, 1))
        _, _, Q, _, _, _, _ = quadraticize(self.cost, xs[N], u, w)
        Qs.append(Q)
        Qbar = sp.linalg.block_diag(*Qs)
        Rubar = sp.linalg.block_diag(*Rus)
        Rwbar = sp.linalg.block_diag(*Rws)

        K = np.zeros((d * N, m * (N + 1)))
        L = np.zeros((n * N, m * (N + 1)))

        for _ in range(self.Tout):
            P_KL = Qbar.copy()
            A_K = Abar - Bbar @ K
            A_L = Abar - Dbar @ L
            for _ in range(self.Tin):
                A_KL = Abar - Bbar @ K - Dbar @ L
                A_L = Abar - Dbar @ L
                P_KL = Qbar.copy()
                for NN in range(N, 0, -1):
                    P_KL = (
                        A_KL.T @ P_KL @ A_KL + Qbar + K.T @ Rubar @ K - L.T @ Rwbar @ L
                    )
                    P_KL = A_KL.T @ P_KL @ A_KL + Qbar + -L.T @ Rwbar @ L
                    if (
                        np.min(np.real(np.linalg.eig(Rwbar - Dbar.T @ P_KL @ Dbar)[0]))
                        < 0
                    ) and self.do_break:
                        raise Exception(f"Eigenvalue condition failed {N - NN}")
                E_KL = (-Rwbar + Dbar.T @ P_KL @ Dbar) @ L - Dbar.T @ P_KL @ A_K
                L = L + self.tau1 * E_KL
            F_KL = (Rubar + Bbar.T @ P_KL @ Bbar) @ K - Bbar.T @ P_KL @ A_L
            K = K - self.tau2 * F_KL

            K = np.nan_to_num(K)
            L = np.nan_to_num(L)

            Qs = []
            Rus = []
            Rws = []
            xn = x.copy()
            for t in range(N):
                u = -K[t * d : (t + 1) * d, t * m : (t + 1) * m] @ xn
                w = -L[t * n : (t + 1) * n, t * m : (t + 1) * m] @ xn
                A, B, D = linearize(self.dynamics, xn, u, w)
                _, _, Q, _, Ru, _, Rw = quadraticize(self.cost, xn, u, w)
                # Rw = 100.0 * np.eye(2)
                Abar[m * (t + 1) : m * (t + 2), m * t : m * (t + 1)] = A
                Bbar[m * (t + 1) : m * (t + 2), d * (t) : d * (t + 1)] = B
                Dbar[m * (t + 1) : m * (t + 2), n * (t) : n * (t + 1)] = D
                Qs.append(Q)
                Rus.append(Ru)
                Rws.append(Rw)
                xn = self.dynamics(xn, u, w)
            _, _, Q, _, _, _, _ = quadraticize(
                self.cost,
                xn,
                np.random.uniform(-1.0, 1.0, (n, 1)),
                np.random.uniform(-1.0, 1.0, (d, 1)),
            )
            Qs.append(Q)
            Qbar = sp.linalg.block_diag(*Qs)
            Rubar = sp.linalg.block_diag(*Rus)
            Rwbar = sp.linalg.block_diag(*Rws)

        us = np.empty((N, d, 1))
        ws = np.empty((N, n, 1))
        xn = x.copy()
        for t in range(N):
            u = -K[t * d : (t + 1) * d, t * m : (t + 1) * m] @ xn
            w = -L[t * n : (t + 1) * n, t * m : (t + 1) * m] @ xn
            us[t] = u
            ws[t] = w
            xn = self.dynamics(xn, u, w)
        return us, ws
