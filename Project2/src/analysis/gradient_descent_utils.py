"""This file contains utility functions used in gradient_descent_plots.py"""
import numpy as np
from collections.abc import Callable

import context
from sknotlearn.data import Data
from sknotlearn.optimize import GradientDescent, SGradientDescent, isNotFinite


def MSE(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    return np.mean((y_pred - y_target)**2)


def OLS_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def ridge_solution(X: np.ndarray, y: np.ndarray, lmbda: float) -> np.ndarray:
    Id = np.eye(X.shape[1])
    return np.linalg.pinv(X.T @ X + lmbda*Id) @ X.T @ y


def clip_exploded_gradients(val: float, lim: float, convergence: int) -> float:
    if lim is not None:
        limtest = val < lim
    else:
        limtest = True
    if convergence and limtest:
        return val
    else:
        return np.nan


def call_optimizer(optimizer: GradientDescent,
                   grad: Callable, x0: np.ndarray,
                   idcs: np.ndarray) -> np.ndarray:
    """Handles the different call signatures of
    GradientDescent and SGradientDescent"""
    if isinstance(optimizer, SGradientDescent):
        theta_opt = optimizer.call(
            grad=grad,
            x0=x0,
            all_idcs=idcs
        )
    else:
        theta_opt = optimizer.call(
            grad=grad,
            x0=x0,
            args=(idcs,)
        )
    return theta_opt


def do_GD_iteration(optimizer: GradientDescent, grad: Callable,
                    data: Data, idcs: np.ndarray) -> None:
    """Do one iteration of gradient descent 'by hand'.

    Args:
        optimizer (GradientDescent): GradientDescent instance to do iteration.
        grad (Callable): Gradient of the relevant cost function to use.
        data (Data): Data instance with data to fit to.
        idcs (np.ndarray): Available indices to pass to grad.
    """
    # Spltting up iterations between stochastic and non-stochastic methods
    if isinstance(optimizer, SGradientDescent):
        batches = optimizer._make_batches(idcs)
        for batch in batches:
            g = grad(optimizer.x, data, batch)
            if isNotFinite(np.abs(g)):
                break
            else:
                optimizer.x = optimizer._update_rule(optimizer, optimizer.x, g)
    else:
        g = grad(optimizer.x, data, idcs)
        if isNotFinite(np.abs(g)):
            pass
        else:
            optimizer.x = optimizer._update_rule(optimizer, optimizer.x, g)
