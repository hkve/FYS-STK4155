import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import context
    from utils import make_figs_path, colors
else:
    from analysis.utils import make_figs_path, colors

from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.resampling import KFold_cross_validate
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data

def plot_train_mse_kfold(degrees, mse_across_folds, title="OLS", filename=None):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(7,5.5))

    alphas = [.75,1,1]
    for i, (k, mses) in enumerate(mse_across_folds.items()):
        train_mse, test_mse = mses
        ax.plot(degrees, train_mse, c=colors[i], alpha=alphas[i], ls="--", label=rf"Train $k = ${k}", lw=2.5)
        ax.plot(degrees, test_mse, c=colors[i], alpha=alphas[i], label=rf"Test $k = ${k}", lw=2.5)

    ks = list(mse_across_folds.keys())
    ax.set_xlabel("Polynomial degrees", fontsize=14)
    ax.set_ylabel(r"$MSE$", fontsize=14)
    ax.legend(fontsize=14)
    ax.set_title(rf"{title} Cross validation, $k \in [{ks[0]},{ks[-1]}]$", fontsize=16)

    fig.tight_layout()
    if filename is not None: plt.savefig(make_figs_path(filename), dpi=300)
    plt.show()

def run_Kfold_cross_validate(Model, degrees, k=5, random_state=321, lmbda=None):    
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
        
    print(f"{Model.__name__}: {k = } Optimal p = {degrees[np.argmin(test_mse)]} with MSE = {np.min(test_mse)}")
    return train_mse, test_mse


if __name__ == "__main__":
    ks = [5,7,10]
    degrees = np.arange(1, 12+1)
    
    # Slicing [:3] is just a very hacky way if you only want to plot some
    Models = [LinearRegression, Ridge, Lasso][:3]
    lmbdas = [None, 0.1, 0.1][:3]
    names =  ["OLS", "Ridge", "Lasso"][:3]

    mse_across_folds = {}

    for Model, lmbda, name in zip(Models, lmbdas, names):
        for k in ks:
            train_mse, test_mse = run_Kfold_cross_validate(Model, degrees, k=k, random_state=321, lmbda=lmbda)
            mse_across_folds[k] = [train_mse, test_mse]
        
        plot_train_mse_kfold(degrees, mse_across_folds, name, filename=f"{name}_mse_kfold")