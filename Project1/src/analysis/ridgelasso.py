from random import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Custom stuff
import context
from sknotlearn.linear_model import Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data
from utils import make_figs_path, colors

def plot_heatmap(degrees_grid, lmbdas_grid, MSEs):
    fig, ax = plt.subplots()

    ax.contourf(degrees_grid, lmbdas_grid, MSEs)
    ax.set_yscale("log")
    plt.show()

def make_mse_grid(Method, degrees, lmbdas):
    train_size = 2/3
    random_state = 321
    noise_std = 0.1
    n = 600
    D = make_FrankeFunction(n=n, uniform=True, random_state=random_state, noise_std=noise_std)

    h, w = len(degrees), len(lmbdas)
    MSEs = np.zeros((h,w))

    for i, degree in enumerate(degrees):
        Dp = D.polynomial(degree=degree)
        Dp = Dp.scaled(scheme="Standard")
        Dp_train, Dp_test = Dp.train_test_split(ratio=train_size, random_state=random_state)

        for j, lmbda in enumerate(lmbdas):
            reg = Method(lmbda=lmbda).fit(Dp_train)
            MSEs[i,j] = reg.mse(Dp_test)

    degrees_grid, lmbdas_grid = np.meshgrid(degrees, lmbdas, indexing="ij")
    return degrees_grid, lmbdas_grid, MSEs


if __name__ == "__main__":
    n_lmbdas = 50
    n_degrees = 20
    lmbdas = np.logspace(-6, 1, n_lmbdas)
    degrees = np.arange(1, n_degrees+1)

    degrees_grid, lmbdas_grid,  MSEs = make_mse_grid(Ridge, degrees, lmbdas)
    plot_heatmap(degrees_grid, lmbdas_grid, MSEs)