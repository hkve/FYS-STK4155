import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plot_utils
import fifa21_utils as utils

Ridge = utils.RidgeInt
Lasso = utils.LassoInt  
LinearRegression = utils.LinearRegressionInt
DecisionTreeRegressor = utils.DecisionTreeRegressorInt


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

def singel_tree_increasing_depth(X, y, filename=None, random_state=321):
    max_depths = np.arange(1,10+1)
    
    c = plot_utils.colors[0]
    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    n = len(max_depths)
    mse, bias, var, regs = utils.bootstrap(X, y, DecisionTreeRegressor, param_name="max_depth", params=max_depths, method_params={"splitter": "best", "random_state": random_state}, save_regs=True, random_state=random_state)

    actual_depth = [reg.get_depth() for reg in regs]
    leaves = [reg.get_n_leaves() for reg in regs]



    fig, ax = plt.subplots()
    ax.plot(max_depths, mse, label="mse", ls=ls["mse"], c=c, marker="*", markersize=8)
    ax.plot(max_depths, bias, label="bias", ls=ls["bias"], c=c, marker="o")
    ax.plot(max_depths, var, label="var", ls=ls["var"], c=c, marker="P")

    ax.set_xticks(max_depths)
    
    ax2 = ax.twiny()
    l = ax.get_xlim()
    l2 = ax2.get_xlim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax.get_xticks())
    ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax2.set_xticklabels([f"{d}/{l}" for d, l in zip(actual_depth, leaves)])

    ax2.grid(False)
    ax.set(xlabel="Allowed depth")
    ax2.set(xlabel="Actual depth/number of leaves")
    ax.legend()
    plot_utils.save(filename)
    plt.show()

if __name__ == "__main__":
    rnd = 3211
    X, y = utils.get_fifa_data(n=10000, random_state=rnd)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=3/4, random_state=rnd)

    # LinearModel_comparison(X_train, y_train, filename="BiasVar_LinearRegression", random_state=rnd)

    singel_tree_increasing_depth(X, y, filename="BiasVar_SingleTree", random_state=rnd)