"""
BSD 3-Clause License

Copyright (c) 2019, HJ Reachability Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author(s): David Fridovich-Keil ( dfk@eecs.berkeley.edu )
"""

"""
Although this code has been modified from the original, we 
include the above copyright notice from the original authors. 

This code is not endorsed by the copyright holder nor the names of its contributors. 
"""


import numpy as np
from collections import deque
from typing import List

from utils.utils import linearize, quadraticize


def solve_lq_game(As, Bs, Qs, ls, Rs):
    horizon = len(As) - 1
    num_players = len(Bs)

    u_dims = [Bis[0].shape[1] for Bis in Bs]

    Zs = [deque([Qis[-1]]) for Qis in Qs]
    zetas = [deque([lis[-1]]) for lis in ls]
    Fs = deque()
    Ps = [deque() for _ in range(num_players)]
    betas = deque()
    alphas = [deque() for _ in range(num_players)]
    for k in range(horizon, -1, -1):
        A = As[k]
        B = [Bis[k] for Bis in Bs]
        Q = [Qis[k] for Qis in Qs]
        l = [lis[k] for lis in ls]
        R = [[Rijs[k] for Rijs in Ris] for Ris in Rs]

        Z = [Zis[0] for Zis in Zs]
        zeta = [zetais[0] for zetais in zetas]

        S_rows = []
        for ii in range(num_players):
            Sis = []
            for jj in range(num_players):
                if jj == ii:
                    Sis.append(R[ii][ii] + B[ii].T @ Z[ii] @ B[ii])
                else:
                    Sis.append(B[ii].T @ Z[ii] @ B[jj])

            S_rows.append(np.concatenate(Sis, axis=1))

        S = np.concatenate(S_rows, axis=0)
        Y = np.concatenate([B[ii].T @ Z[ii] @ A for ii in range(num_players)], axis=0)

        P, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        P_split = np.split(P, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            Ps[ii].appendleft(P_split[ii])

        F = A - sum([B[ii] @ P_split[ii] for ii in range(num_players)])
        Fs.appendleft(F)

        for ii in range(num_players):
            Zs[ii].appendleft(
                F.T @ Z[ii] @ F
                + Q[ii]
                + sum(
                    [
                        P_split[jj].T @ R[ii][jj] @ P_split[jj]
                        for jj in range(num_players)
                    ]
                )
            )

        Y = np.concatenate([B[ii].T @ zeta[ii] for ii in range(num_players)], axis=0)

        alpha, _, _, _ = np.linalg.lstsq(a=S, b=Y, rcond=None)
        alpha_split = np.split(alpha, np.cumsum(u_dims[:-1]), axis=0)
        for ii in range(num_players):
            alphas[ii].appendleft(alpha_split[ii])

        beta = -sum([B[ii] @ alpha_split[ii] for ii in range(num_players)])
        betas.appendleft(beta)

        for ii in range(num_players):
            zetas[ii].appendleft(
                F.T @ (zeta[ii] + Z[ii] @ beta)
                + l[ii]
                + sum(
                    [
                        P_split[jj].T @ R[ii][jj] @ alpha_split[jj]
                        for jj in range(num_players)
                    ]
                )
            )

    return [list(Pis) for Pis in Ps], [list(alphais) for alphais in alphas]


class ILQSolver(object):
    def __init__(
        self, env, alpha_scaling=0.001, clipa=-1.0, dobreak=False, horizon: int = 10
    ):

        self.dynamics = env.dynamics
        self.agent1_cost = env.robot_cost
        self.agent2_cost = lambda *args: -env.robot_cost(*args)
        self.agent1_Ps = [
            np.zeros((env.udim, env.xdim * env.n_agents)) for _ in range(horizon)
        ]
        self.agent2_Ps = [
            np.zeros((env.wdim, env.xdim * env.n_agents)) for _ in range(horizon)
        ]
        self.agent1_alphas = [np.zeros((env.udim, 1)) for _ in range(horizon)]
        self.agent2_alphas = [np.zeros((env.wdim, 1)) for _ in range(horizon)]

        self.horizon = horizon
        self.xdim = len(env.x)
        self.udim = env.udim
        self.wdim = env.wdim
        self.alpha_scaling = alpha_scaling

        self.last_operating_point = None
        self.current_operating_point = None

        self.clip = clipa > 0.0
        if self.clip:
            self.clipa = clipa

        self.dobreak = dobreak

    def compute_operating_point(self):
        xs = [self.x0]
        us = []
        ws = []
        for t in range(self.horizon):
            if self.current_operating_point is not None:
                xn = self.current_operating_point[0][t]
                un = self.current_operating_point[1][t]
                wn = self.current_operating_point[2][t]
            else:
                xn = np.zeros((self.xdim, 1))
                un = np.zeros((self.udim, 1))
                wn = np.zeros((self.wdim, 1))

            feedback = (
                lambda x, uref, xref, P, alpha: uref
                - P @ (x - xref)
                - self.alpha_scaling * alpha
            )
            u = feedback(xs[t], un, xn, self.agent1_Ps[t], self.agent1_alphas[t])
            w = feedback(xs[t], wn, xn, self.agent2_Ps[t], self.agent2_alphas[t])
            if self.clip:
                u = np.clip(
                    u,
                    -self.clipa,
                    self.clipa,
                )
                w = np.clip(
                    w,
                    -self.clipa,
                    self.clipa,
                )
            us.append(u)
            ws.append(w)
            xs.append(self.dynamics(xs[t], u, w))

        return xs, us, ws

    def is_converged(self, TOL=1e-4):
        if not self.dobreak:
            return False
        if self.last_operating_point is None or self.current_operating_point is None:
            return False
        for t in range(self.horizon):
            last_x = self.last_operating_point[0][t]
            current_x = self.current_operating_point[0][t]
            if np.linalg.norm(last_x - current_x) > TOL:
                return False
        return True

    def solve(self, x, iterations=100, TOL=1e-9):
        """
        the real meat of the problem
        """
        self.x0 = x
        iteration = 0
        while not self.is_converged(TOL=TOL) and iteration < iterations:
            xs, us, ws = self.compute_operating_point()
            self.last_operating_point = self.current_operating_point
            self.current_operating_point = [xs, us, ws]

            As = []
            Bs = []
            Ds = []
            for t in range(self.horizon):
                A, B, D = linearize(self.dynamics, xs[t], us[t], ws[t])
                As.append(A)
                Bs.append(B)
                Ds.append(D)

            Qus = []
            Qws = []
            lus = []
            lws = []
            Ru1s = []
            Rw1s = []
            Ru2s = []
            Rw2s = []
            for t in range(self.horizon):
                _, lx1, Qx1, _, Ru1, _, Rw1 = quadraticize(
                    self.agent1_cost, xs[t], us[t], ws[t]
                )
                _, lx2, Qx2, _, Ru2, _, Rw2 = quadraticize(
                    self.agent2_cost, xs[t], us[t], ws[t]
                )
                Qus.append(Qx1)
                Qws.append(Qx2)
                lus.append(lx1)
                lws.append(lx2)
                Ru1s.append(Ru1)
                Rw1s.append(Rw1)
                Ru2s.append(Ru2)
                Rw2s.append(Rw2)

            Ps, alphas = solve_lq_game(
                As, [Bs, Ds], [Qus, Qws], [lus, lws], [[Ru1s, Rw1s], [Ru2s, Rw2s]]
            )
            self.agent1_Ps = Ps[0]
            self.agent2_Ps = Ps[1]
            self.agent1_alphas = alphas[0]
            self.agent2_alphas = alphas[1]
            iteration += 1
        _, us, ws = self.current_operating_point  # type: ignore
        return us, ws
