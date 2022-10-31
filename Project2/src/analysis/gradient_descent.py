import autograd.numpy as np
from autograd import elementwise_grad
import matplotlib.pyplot as plt
import sys
from collections.abc import Callable
import plot_utils
from utils import make_figs_path

import context
from sknotlearn.data import Data
from sknotlearn.linear_model import OLS_gradient, ridge_gradient
from sknotlearn.optimize import GradientDescent, SGradientDescent
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction


def tune_learning_rate(
    data:Data,
    ratio:float,
    learning_rates:np.ndarray,
    optimizers:tuple,
    optimizer_names:tuple,
    cost:str = "OLS",
    lmbda:float=None,
    ylims = (0,3),
    filename=None,
    random_state:int = None
) -> None:

    if random_state:
        np.random.seed(random_state)

    data_train, data_test = data.train_test_split(ratio=ratio, random_state=random_state)
    y_train, X_train = data_train.unpacked()
    y_test, X_test = data_test.unpacked()

    theta0 = np.random.randn(X_test.shape[1])

    if cost == "OLS":
        grad = lambda theta, idcs : OLS_gradient(theta, data_train, idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train) @ X_train.T @ y_train
    elif cost == "ridge":
        grad = lambda theta, idcs : ridge_gradient(theta, data_train, lmbda, idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train + lmbda*np.eye(X_train.shape[1])) @ X_train.T @ y_train


    for optimizer, name in zip(optimizers, optimizer_names):
        MSE_list = list()
        for learning_rate in learning_rates:
            params = optimizer.params
            params["eta"] = learning_rate
            optimizer.set_params(params)

            if isinstance(optimizer, SGradientDescent):
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    all_idcs=np.arange(len(data_train))
                )
            else:
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    args=(np.arange(len(data_train)),)
                )

            
            MSE_list.append(np.mean((X_test@theta_opt - y_test)**2))
        plt.plot(learning_rates, MSE_list, label=name)

    MSE_ana = np.mean((X_test@theta_ana - y_test)**2)
    print(MSE_ana)
    plt.hlines(MSE_ana,
        xmin=learning_rates.min(),
        xmax=learning_rates.max(),
        label=f"Analytical {cost} solution",
        ls="--",
        colors="gray"
    )
    plt.ylim(ylims)
    plt.xlabel("Learning rate")
    plt.ylabel("Validation MSE")
    plt.legend(loc="upper right")
    if filename:
        plt.savefig(make_figs_path(filename))
    plt.show()


if __name__=="__main__":
    random_state = 321
    D = make_FrankeFunction(n=600, noise_std=0.1, random_state=random_state)
    D = D.polynomial(degree=5).scaled(scheme="Standard")

    learning_rates = np.linspace(1e-3, 0.1, 30, endpoint=False)

    GD = GradientDescent("plain", {"eta":0.1}, its=100)
    mGD = GradientDescent("momentum", {"gamma":0.1, "eta":0.1}, its=100)
    SGD = SGradientDescent("plain", {"eta":0.1}, epochs=100, batch_size=54, random_state=random_state)

    tune_learning_rate(
        data=D,
        ratio=3/4,
        learning_rates=learning_rates,
        optimizers=(GD,mGD,SGD),
        optimizer_names=("Plain GD","Momentum GD", "Plain SGD"),
        cost="OLS",
        ylims=(0,3),
        random_state=random_state
    )