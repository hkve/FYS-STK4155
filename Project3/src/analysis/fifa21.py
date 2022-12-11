import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plot_utils
import fifa21_utils as utils

Ridge = utils.RidgeInt
Lasso = utils.LassoInt  
LinearRegression = utils.LinearRegressionInt
import matplotlib.ticker as mticker

def LinearModel_comparison(X, y, filename=None, random_state=321):
    alpha = np.logspace(-5, np.log10(5), 100)
    
    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    c = {
        Ridge: plot_utils.colors[0],
        Lasso: plot_utils.colors[1],
        LinearRegression: plot_utils.colors[2]
    }

    method_params = {Ridge: {"positive": True}, Lasso: {"max_iter": 10000, "positive": True}}
    
    mins = {}

    fig, ax = plt.subplots()
    for i, Reg in enumerate([Ridge, Lasso]):
        mse, bias, var = utils.bootstrap(X_train, y_train, Reg, "alpha", alpha, method_params=method_params[Reg], random_state=rnd)
        
        ax.plot(alpha, mse, label="mse", ls=ls["mse"], c=c[Reg])
        ax.plot(alpha, bias, label="bias", ls=ls["bias"], c=c[Reg])
        ax.plot(alpha, var, label="var", ls=ls["var"], c=c[Reg])
        
        i_min = np.argmin(mse)
        mins[Reg] = {"alpha": alpha[i_min], "mse": mse[i_min]} 
    
    mse, bias, var = utils.boostrap_single(X_train, y_train, method=LinearRegression, method_params={"positive": True}, random_state=rnd)
    ax.scatter(1e-5, mse, label="mse", marker="*", color=c[LinearRegression])
    ax.scatter(1e-5, bias, label="bias", marker="o", color=c[LinearRegression])
    ax.scatter(1e-5, var, label="var", marker="P", color=c[LinearRegression])

    ax.set_yscale("log")
    ax.set_xscale("log")
    old_lim = ax.get_ylim()
    new_lim = (old_lim[0], old_lim[1]*1.4)
    
    x = [mins[Ridge]["alpha"], mins[Lasso]["alpha"]]
    y = [mins[Ridge]["mse"], mins[Lasso]["mse"]]
    ax.vlines(x=x, ymin=new_lim[0], ymax=y, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x, y, color="k", marker="x", alpha=0.25)
    ax.legend(ncol=3)
    ax.set(xlabel=r"Regularisation $\alpha$", ylim=new_lim)

    plot_utils.save(filename)
    plt.show()



if __name__ == "__main__":
    rnd = 3211
    X, y = utils.get_fifa_data(n=10000, random_state=rnd)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=3/4, random_state=rnd)

    LinearModel_comparison(X_train, y_train, filename="BiasVar_LinearRegression", random_state=rnd)