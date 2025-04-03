import numpy as np


def human_model(
    env,
    x,
    u,
    wseed: np.ndarray | None = None,
    T: int = 1,
    N: int = 200,
    lb: float | np.ndarray = -1.0,
    ub: float | np.ndarray = 1.0,
    beta: float = 5.0,
):
    if wseed is None:
        wseed = np.zeros((T, env.wdim, 1))
    search_space = np.random.uniform(lb, ub, (N, T, env.wdim, 1)) + wseed
    search_space[0] *= 0.0
    search_space = search_space.clip(lb, ub)
    weights = np.empty(N)
    for i in range(N):
        w = search_space[i]
        c = 0.0
        xn = x.copy()
        for t in range(T):
            wt = w[t]
            c += env.human_cost(xn, u, wt)
            xn = env.dynamics(xn, u, wt)
        weights[i] = np.exp(-beta * c)
    weights /= np.sum(weights)
    return search_space[np.random.choice(np.arange(N), p=weights)]
