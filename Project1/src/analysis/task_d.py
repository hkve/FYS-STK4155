import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.resampling import KFold_cross_validate
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data

from task_b import plot_train_test


def task_d(n=600, k=5, random_state=321):    
    degrees = np.arange(1, 12)
    train_mse = np.zeros_like(degrees, dtype=float)
    test_mse = np.zeros_like(degrees, dtype=float)

    D = make_FrankeFunction(n, noise_std=0.1, random_state=random_state)

    for i, degree in enumerate(degrees):
        Dp = D.polynomial(degree=degree)
        Dp = Dp.scaled(scheme="Standard")

        reg = LinearRegression()
        resampler = KFold_cross_validate(reg, Dp, k=k, scoring="mse", random_state=random_state)

        train_mse[i] = resampler["train_mse"].mean()
        test_mse[i] = resampler["test_mse"].mean()
        

    return degrees, train_mse, test_mse
if __name__ == "__main__":
    ks = [5,7,10]
    filename = "OLS_mse_kfoldcross"

    for k in ks:
        degrees, train_mse, test_mse = task_d(k=k, random_state=321)
        plot_train_test(degrees, train_mse, test_mse, filename=f"{filename}{k}", title=f"OLS k = {k} Cross-validation")