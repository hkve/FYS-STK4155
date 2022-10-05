import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Custom stuff
if __name__ == "__main__":
    import context
    from utils import make_figs_path, colors
else:
    from analysis.utils import make_figs_path, colors
    
from sknotlearn.linear_model import Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.resampling import KFold_cross_validate
from sknotlearn.data import Data

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


def min_params(degrees_grid, lmbdas_grid, MSEs):
    best_idx = np.unravel_index(MSEs.argmin(), MSEs.shape)
    best_idxs = np.unravel_index(MSEs.argmin(axis=1), MSEs.shape)
    return degrees_grid[best_idx], lmbdas_grid[best_idx], lmbdas_grid[best_idxs]


def plot_heatmap(degrees_grid, lmbdas_grid, MSEs, model=None, filename=None):
    n_levels = 50
    
    levels = np.linspace(np.min(MSEs), np.max(MSEs), n_levels)

    best_degree, best_lmbda, best_lmbdas = min_params(degrees_grid, lmbdas_grid, MSEs)

    fig, ax = plt.subplots()

    cont = ax.contourf(degrees_grid, lmbdas_grid, MSEs, levels=levels, cmap="viridis")
    ax.scatter(degrees_grid[:,0], best_lmbdas, marker="x", s=40, color="r", alpha=0.4)
    ax.scatter(best_degree, best_lmbda, marker="X", s=160, color="r", alpha=0.6)
    ax.text(best_degree-1.7, best_lmbda*1.7, f"$({best_degree:n}, {best_lmbda:.0E})$", color="w")
    cbar = fig.colorbar(cont, pad=0.01, aspect=6)
    cbar.set_label("MSE", fontsize=14)

    if model:
        ax.set_title(f"Optimisation of {model} using CV.", fontsize=16)
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree", fontsize=14)
    ax.set_ylabel(r"Log$_{10}(\lambda)$", fontsize=14)
    fig.tight_layout()

    if filename:
        plt.savefig(make_figs_path(filename), dpi=300, bbox_inches="tight")

    plt.show()


def make_mse_grid(Method, D, degrees, lmbdas, train_size=2/3, random_state=321):
    h, w = len(degrees), len(lmbdas)
    MSEs = np.zeros((h,w))

    for i, degree in enumerate(degrees):
        Dp = D.polynomial(degree=degree).scaled(scheme="Standard")

        for j, lmbda in enumerate(lmbdas):
            resampler = KFold_cross_validate(
                reg = Method(lmbda=lmbda),
                data = Dp,
                k = 7,
                scoring = ("mse"),
                shuffle = False,
                run=True,
                random_state=321,
            )
            MSEs[i,j] = np.mean(resampler["test_mse"])

    degrees_grid, lmbdas_grid = np.meshgrid(degrees, lmbdas, indexing="ij")
    return degrees_grid, lmbdas_grid, MSEs


if __name__ == "__main__":
    n_lmbdas = 21
    n_degrees = 15
    lmbdas = np.logspace(-9, 1, n_lmbdas)
    degrees = np.arange(1, n_degrees+1)

    train_size = 2/3
    random_state = 321
    noise_std = 0.1
    n = 600

    D = make_FrankeFunction(n=n, linspace=False, random_state=random_state, noise_std=noise_std)

    # Ridge
    params = make_mse_grid(Ridge, D, degrees, lmbdas, train_size, random_state)
    # dump("ridge_grid", *params)
    # params = load("ridge_grid")
    plot_heatmap(*params, model="Ridge", filename="heatmap_ridge")

    # Lasso
    # params = make_mse_grid(Lasso, D, degrees, lmbdas, train_size, random_state)
    # dump("lasso_grid", *params)
    # params = load("lasso_grid")
    # plot_heatmap(*params, model="Lasso", filename="heatmap_lasso")