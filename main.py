import tqdm
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from method_ilq import ILQSolver
from method_mclq import MCLQSolver
from method_npg import NPGSolver
from envs.driving import Env as DrivingEnv
from envs.pointmass import Env as PointMassEnv
from utils.human import human_model


def main_liveplot(env, method, args):
    fig = plt.figure()
    plt.ion()
    tbar = tqdm.trange(args.trials, desc="Trials")
    for _ in tbar:
        fig.clear()
        cr = 0.0
        ch = 0.0
        plt.subplot(1, 2, 1)
        plt.xlim([-20.0, 20.0])
        plt.ylim([-20.0, 20.0])
        plt.plot(*env.u_goal.flatten(), "b*")
        plt.plot(*env.w_goal.flatten(), "r*")
        for t in range(args.timesteps):
            us, _ = method.solve(env.x)
            u = us[0].clip(-1.0, 1.0)
            ws = human_model(
                env, env.x, np.zeros((env.udim, 1)), np.zeros((env.wdim, 1))
            )
            w = ws[0]
            cr += env.robot_cost(env.x, u, w)
            ch += env.human_cost(env.x, u, w)
            # plotting
            plt.subplot(1, 2, 1)
            plt.xlim([-20.0, 20.0])
            plt.ylim([-20.0, 20.0])
            if args.env == "pointmass":
                plt.plot(*env.x[0:2].flatten(), "b.")
                plt.plot(*env.x[2:4].flatten(), "r.")
            elif args.env == "driving":
                plt.plot(*env.x[0:2].flatten(), "b.")
                plt.plot(*env.x[4:6].flatten(), "r.")
            plt.subplot(1, 2, 2)
            plt.plot(t, cr, "b.")
            plt.plot(t, ch, "r.")
            plt.pause(0.1)
            env.step(u, w)
        tbar.set_description(f"method {args.method} in {args.env}: c={cr:2.2f}")
    tbar.close()
    return


def main(args):
    if args.env == "driving":
        env = DrivingEnv(
            xlb=-10.0,
            xub=-10.0,
        )
    else:
        env = PointMassEnv()
    if args.method == "mclq":
        method = MCLQSolver(env, horizon=args.horizon)
    elif args.method == "npg":
        method = NPGSolver(env, horizon=args.horizon)
    else:
        method = ILQSolver(env, horizon=args.horizon)
    if args.live_plot:
        return main_liveplot(env, method, args)
    tbar = tqdm.trange(args.trials, desc="Trials")
    for _ in tbar:
        c = 0.0
        for _ in range(args.timesteps):
            us, _ = method.solve(env.x)
            u = us[0].clip(-1.0, 1.0)
            ws = human_model(
                env, env.x, np.zeros((env.udim, 1)), np.zeros((env.wdim, 1))
            )
            w = ws[0]
            c += env.robot_cost(env.x, u, w)
            env.step(u, w)
        tbar.set_description(f"method {args.method} in {args.env}: c={c:2.2f}")
    tbar.close()
    return


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    parser = ArgumentParser()
    parser.add_argument("--method", choices=["npg", "ilq", "mclq"], default="mclq")
    parser.add_argument("--env", choices=["driving", "pointmass"], default="driving")
    parser.add_argument("--live-plot", action="store_true", default=False)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    main(args)
