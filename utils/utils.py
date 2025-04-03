import torch
import numpy as np

from typing import Callable


def quadraticize(J: Callable, x: np.ndarray, u: np.ndarray, w: np.ndarray):
    x_torch = torch.from_numpy(x).requires_grad_(True)
    u_torch = torch.from_numpy(u).requires_grad_(True)
    w_torch = torch.from_numpy(w).requires_grad_(True)

    xdim = len(x)
    udim = len(u)
    wdim = len(w)

    cost_torch = J(x_torch, u_torch, w_torch)
    cost = cost_torch.item()

    hess_x = np.zeros((xdim, xdim))
    grad_x = np.zeros((xdim, 1))

    grad_x_torch = torch.autograd.grad(
        cost_torch, x_torch, create_graph=True, allow_unused=True
    )[0]
    if grad_x_torch is not None:
        grad_x = grad_x_torch.detach().numpy().copy()
        for ii in range(xdim):
            hess_row = torch.autograd.grad(
                grad_x_torch[ii, 0],
                x_torch,
                retain_graph=True,
                allow_unused=True,
            )[0]
            hess_x[ii, :] = hess_row.detach().numpy().copy().T

    grad_u_torch = torch.autograd.grad(
        cost_torch, u_torch, create_graph=True, allow_unused=True
    )[0]
    hess_u = np.zeros((udim, udim))
    grad_u = np.zeros((udim, 1))
    if grad_u_torch is not None:
        grad_u = grad_u_torch.detach().numpy().copy()
        for ii in range(udim):
            hess_u_row = torch.autograd.grad(
                grad_u_torch[ii, 0],
                u_torch,
                retain_graph=True,
                allow_unused=True,
            )[0]
            hess_u[ii, :] = hess_u_row.detach().numpy().copy().T

    grad_w_torch = torch.autograd.grad(
        cost_torch, w_torch, create_graph=True, allow_unused=True
    )[0]

    hess_w = np.zeros((wdim, wdim))
    grad_w = np.zeros((wdim, 1))
    if grad_w_torch is not None:
        grad_w = grad_w_torch.detach().numpy().copy()
        for ii in range(wdim):
            hess_w_row = torch.autograd.grad(
                grad_w_torch[ii, 0],
                w_torch,
                retain_graph=True,
                allow_unused=True,
            )[0]
            hess_w[ii, :] = hess_w_row.detach().numpy().copy().T

    grad_x = np.nan_to_num(grad_x)
    hess_x = np.nan_to_num(hess_x)
    grad_u = np.nan_to_num(grad_u)
    hess_u = np.nan_to_num(hess_u)
    grad_w = np.nan_to_num(grad_w)
    hess_w = np.nan_to_num(hess_w)

    return cost, grad_x, hess_x, grad_u, hess_u, grad_w, hess_w


def linearize(f: Callable, x: np.ndarray, u: np.ndarray, w: np.ndarray):
    x_torch = torch.from_numpy(x).requires_grad_(True)
    u_torch = torch.from_numpy(u).requires_grad_(True)
    w_torch = torch.from_numpy(w).requires_grad_(True)
    xn = f(x_torch, u_torch, w_torch)
    x_gradient_list = []
    u_gradient_list = []
    w_gradient_list = []
    xdim = len(x)
    for ii in range(xdim):
        x_gradient_list.append(
            torch.autograd.grad(xn[ii, 0], x_torch, retain_graph=True)[0]
        )
        u_gradient_list.append(
            torch.autograd.grad(xn[ii, 0], u_torch, retain_graph=True)[0]
        )
        w_gradient_list.append(
            torch.autograd.grad(xn[ii, 0], w_torch, retain_graph=True)[0]
        )

    A = torch.cat(x_gradient_list, dim=1).detach().numpy().copy().T
    B = torch.cat(u_gradient_list, dim=1).detach().numpy().copy().T
    D = torch.cat(w_gradient_list, dim=1).detach().numpy().copy().T
    return A, B, D
