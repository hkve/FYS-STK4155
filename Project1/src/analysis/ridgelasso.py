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

def dump(filename, degrees_grid, lmbdas_grid, MSEs):
    with open(f"{filename}.npz", "wb") as fp: 
        np.save(fp, degrees_grid)
        np.save(fp, lmbdas_grid)
        np.save(fp, MSEs)
    

def load(filename):
    with open(f"{filename}.npz", "rb") as fp:
        degrees_grid = np.load(fp)
        lmbdas_grid = np.load(fp)
        MSEs = np.load(fp)

    return degrees_grid, lmbdas_grid, MSEs

def plot_heatmap(degrees_grid, lmbdas_grid, MSEs):
    n_levels = 50
    
    levels = np.linspace(np.min(MSEs), np.max(MSEs), n_levels)

    fig, ax = plt.subplots()

    cont = ax.contourf(degrees_grid, lmbdas_grid, MSEs, levels=levels, cmap="viridis")
    cbar = fig.colorbar(cont)

    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree", fontsize=14)
    ax.set_ylabel(r"Log$_{10}(\lambda)$", fontsize=14)
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
            # Add resampling option here
            reg = Method(lmbda=lmbda).fit(Dp_train)
            MSEs[i,j] = reg.mse(Dp_test)

    degrees_grid, lmbdas_grid = np.meshgrid(degrees, lmbdas, indexing="ij")
    return degrees_grid, lmbdas_grid, MSEs


if __name__ == "__main__":
    n_lmbdas = 20
    n_degrees = 20
    lmbdas = np.logspace(-6, 1, n_lmbdas)
    degrees = np.arange(1, n_degrees+1)

    # Ridge
    params = make_mse_grid(Ridge, degrees, lmbdas)
    dump("ridge_grid2", *params)
    params = load("ridge_grid2")
    plot_heatmap(*params)

    # Lasso
    # params = make_mse_grid(Lasso, degrees, lmbdas)
    # dump("lasso_grid", *params)
    # params = load("lasso_grid")
    # plot_heatmap(*params)