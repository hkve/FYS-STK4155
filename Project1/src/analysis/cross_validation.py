import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import context
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.resampling import KFold_cross_validate
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data

from noresampling import plot_train_test


def run_Kfold_cross_validate(Model, degrees, n=600, k=5, random_state=321, lmbda=None):    
    train_mse = np.zeros_like(degrees, dtype=float)
    test_mse = np.zeros_like(degrees, dtype=float)

    D = make_FrankeFunction(n, noise_std=0.1, random_state=random_state)

    for i, degree in enumerate(degrees):
        Dp = D.polynomial(degree=degree)
        Dp = Dp.scaled(scheme="Standard")

        if Model in [Ridge, Lasso]:
            assert lmbda is not None
            reg = Model(lmbda = lmbda)
        else:
            reg = Model()

        resampler = KFold_cross_validate(reg, Dp, k=k, scoring="mse", random_state=random_state)

        train_mse[i] = resampler["train_mse"].mean()
        test_mse[i] = resampler["test_mse"].mean()
        

    return train_mse, test_mse


if __name__ == "__main__":
    ks = [5,7,10]
    degrees = np.arange(1, 12+1)

    # Slicing [:3] is just a very hacky way if you only want to plot some
    Models = [LinearRegression, Ridge, Lasso][:3]
    lmbdas = [None, 0.1, 0.1][:3]
    names =  ["OLS", "Ridge", "Lasso"][:3]

    for Model, lmbda, name in zip(Models, lmbdas, names):
        for k in ks:
            train_mse, test_mse = run_Kfold_cross_validate(Model, degrees, k=k, random_state=321, lmbda=lmbda)
            plot_train_test(degrees, train_mse, test_mse, filename=f"{name}_mse_kfoldcross_k{k}", title=f"{name} k = {k} Cross-validation")