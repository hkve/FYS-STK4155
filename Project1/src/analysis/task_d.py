import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.resampling import KFold_cross_validate
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data

def plot_mse(degrees, train_mse, test_mse):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()

    ax.plot(degrees, train_mse, label="Train mse")
    ax.plot(degrees, test_mse, label="Test mse")    
    ax.legend()

    plt.show()

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
        
    plot_mse(degrees, train_mse, test_mse)

if __name__ == "__main__":
    task_d(random_state=321)