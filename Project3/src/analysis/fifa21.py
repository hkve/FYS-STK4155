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
BaggingRegressor = utils.BaggingRegressorInt
GradientBoostingRegressor = utils.GradientBoostingRegressorInt
SVR = utils.SVRInt

def LinearModel(X, y, filename=None, random_state=321):
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
        
        ax.plot(alpha, mse, label="MSE", ls=ls["mse"], c=c[Reg])
        ax.plot(alpha, bias, label="Bias$^2$", ls=ls["bias"], c=c[Reg])
        ax.plot(alpha, var, label="Var", ls=ls["var"], c=c[Reg])
        
        i_min = np.argmin(mse)
        mins[Reg] = {"alpha": alpha[i_min], "mse": mse[i_min]} 
    
    mse, bias, var = utils.boostrap_single(X_train, y_train, method=LinearRegression, method_params={"positive": True}, random_state=rnd)
    ax.scatter(1e-5, mse, label="MSE", marker="*", color=c[LinearRegression])
    ax.scatter(1e-5, bias, label="Bias$^2$", marker="o", color=c[LinearRegression])
    ax.scatter(1e-5, var, label="Var", marker="P", color=c[LinearRegression])

    ax.set_yscale("log")
    ax.set_xscale("log")
    old_lim = ax.get_ylim()
    new_lim = (old_lim[0], old_lim[1]*1.4)
    
    x = [mins[Ridge]["alpha"], mins[Lasso]["alpha"]]
    y = [mins[Ridge]["mse"], mins[Lasso]["mse"]]
    print(mins)
    ax.vlines(x=x, ymin=new_lim[0], ymax=y, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x, y, color="k", marker="x", alpha=0.25)
    ax.legend(ncol=3)
    ax.set(xlabel=r"Regularisation $\alpha$", ylim=new_lim)

    plot_utils.save(filename)
    plt.show()

def Singel_tree_increasing_depth(X, y, filename=None, random_state=321):
    max_depths = np.arange(1,10+1)
    
    c = plot_utils.colors[0]
    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    n = len(max_depths)
    mse, bias, var, regs = utils.bootstrap(X, y, DecisionTreeRegressor, param_name="max_depth", params=max_depths, method_params={"splitter": "best", "random_state": random_state}, save_regs=True, random_state=random_state)

    train_mse = np.zeros_like(mse)
    for i, reg in enumerate(regs):
        train_mse[i] = reg.train_mse

    actual_depth = [reg.get_depth() for reg in regs]
    leaves = [reg.get_n_leaves() for reg in regs]



    fig, ax = plt.subplots()
    ax.plot(max_depths, train_mse, label="Train MSE", c=plot_utils.colors[1], alpha=0.5)
    ax.plot(max_depths, mse, label="MSE", ls=ls["mse"], c=c, marker="*", markersize=8)
    ax.plot(max_depths, bias, label="Bias$^2$", ls=ls["bias"], c=c, marker="o")
    ax.plot(max_depths, var, label="Var", ls=ls["var"], c=c, marker="P")

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
    old_ylims = ax.get_ylim()
    ax.set(ylim=(old_ylims[0], old_ylims[1]*1.2))
    ax.legend(ncol=4, loc="upper center")
    plot_utils.save(filename)
    plt.show()

def Trees_increasing_ensamble(X, y, filename=None, random_state=321):
    n_estimators = np.linspace(2,100,51, dtype=int)
    
    n = len(n_estimators)
    mse, bias, var = np.zeros(n), np.zeros(n), np.zeros(n)
    c = plot_utils.colors[0]
    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    c = {
        "Bag": plot_utils.colors[0],
        "Rf": plot_utils.colors[1]
    }

    fig, ax = plt.subplots()
    max_features = {
        "Bag": 1.0,
        "Rf": 0.33
    }

    mins = {}

    for method in ["Bag", "Rf"]:
        for i, n in enumerate(n_estimators):
            reg = BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    max_depth=10,
                    random_state=random_state
                ),
                max_features=max_features[method],
                n_estimators=n,
                random_state=random_state
            )

            mse[i], bias[i], var[i] = utils.bv_decomp(X,y,reg,random_state=random_state)
        
        mins[method] = {"n": n_estimators[np.argmin(mse)], "mse": np.min(mse)}

        ax.plot(n_estimators, mse, label="MSE", ls=ls["mse"], c=c[method])
        ax.plot(n_estimators, bias, label="Bias$^2$", ls=ls["bias"], c=c[method])
        ax.plot(n_estimators, var, label="Var", ls=ls["var"], c=c[method])

    x = [mins["Bag"]["n"], mins["Rf"]["n"]]
    y = [mins["Bag"]["mse"], mins["Rf"]["mse"]]
    ax.vlines(x=x, ymin=ax.get_ylim()[0], ymax=y, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x, y, color="k", marker="x", alpha=0.25)
    ax.set_xlabel("Ensamble size")
    ax.legend(ncol=2)
    plot_utils.save(filename)
    plt.show()

def Boosting(X, y, filename=None, random_state=321):
    n_estimators = np.linspace(10,100,51, dtype=int)

    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }
    c = plot_utils.colors[0]

    mse, bias, var = utils.bootstrap(X, y, GradientBoostingRegressor, param_name="n_estimators", params=n_estimators, method_params={"criterion": "squared_error"})

    fig, ax = plt.subplots()
    ax.plot(n_estimators, mse, label="MSE", ls=ls["mse"], c=c)
    ax.plot(n_estimators, bias, label="Bias$^2$", ls=ls["bias"], c=c)
    ax.plot(n_estimators, var, label="Var", ls=ls["var"], c=c)
    ax.set(xlabel="Ensamble size")
    ax.legend(ncol=3)

    y_mse = np.min(mse)
    x_n_est = n_estimators[np.argmin(mse)]
    ax.vlines(x=x_n_est, ymin=ax.get_ylim()[0], ymax=y_mse, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x_n_est, y_mse, color="k", marker="x", alpha=0.25)

    plot_utils.save(filename + "_n_est")
    plt.show()

    subsamples = np.linspace(0.2,0.9, 8)

    mse, bias, var = utils.bootstrap(X, y, GradientBoostingRegressor, param_name="subsample", params=subsamples, method_params={"criterion": "squared_error", "n_estimators": 40})

    fig, ax = plt.subplots()
    ax.plot(subsamples, mse, label="MSE", ls=ls["mse"], c=c)
    ax.plot(subsamples, bias, label="Bias$^2$", ls=ls["bias"], c=c)
    ax.plot(subsamples, var, label="Var", ls=ls["var"], c=c)

    ax.legend(ncol=3)
    plot_utils.save(filename + "_subsamples")
    plt.show()

def SupperVecReg(X, y, filename=None, random_state=321):
    eps = np.logspace(-3, 0, 100)
    
    kernels = ["linear", "rbf"]

    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    c = {kernel: plot_utils.colors[i] for i, kernel in enumerate(kernels)}
    fig, ax = plt.subplots()

    mins = {kernel: {} for kernel in kernels}
    for kernel in kernels:
        method_params = {"kernel": kernel}
        mse, bias, var = utils.bootstrap(X, y, SVR, param_name="epsilon", params=eps, method_params=method_params, scale_y=True)

        mins[kernel]["mse"] = np.min(mse)
        mins[kernel]["eps"] = eps[np.argmin(mse)]

        ax.plot(eps, mse, label="MSE", ls=ls["mse"], c=c[kernel])
        ax.plot(eps, bias, label="Bias$^2$", ls=ls["bias"], c=c[kernel])
        ax.plot(eps, var, label="Var", ls=ls["var"], c=c[kernel])
    
    print(mins)
    x_eps, y_mse = [], []
    for v in mins.values():
        x_eps.append(v["eps"])
        y_mse.append(v["mse"])

    ax.vlines(x=x_eps, ymin=ax.get_ylim()[0], ymax=y_mse, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x_eps, y_mse, color="k", marker="x", alpha=0.25)
    ax.set_xscale("log")
    ax.set(xlabel="$\epsilon$")
    ax.legend(ncol=2)
    # plot_utils.save(filename + "eps")
    plt.show()

    fig, ax = plt.subplots()
    C = np.logspace(-1, 2, 100)
    pen_mins = {kernel: {} for kernel in kernels}
    for kernel in kernels:
        method_params = {"kernel": kernel, "epsilon": mins[kernel]["eps"]}
        mse, bias, var = utils.bootstrap(X, y, SVR, param_name="C", params=C, method_params=method_params, scale_y=True)

        pen_mins[kernel]["mse"] = np.min(mse)
        pen_mins[kernel]["C"] = C[np.argmin(mse)]

        ax.plot(C, mse, label="MSE", ls=ls["mse"], c=c[kernel])
        ax.plot(C, bias, label="Bias$^2$", ls=ls["bias"], c=c[kernel])
        ax.plot(C, var, label="Var", ls=ls["var"], c=c[kernel])
    
    print(pen_mins)
    x_C, y_mse = [], []
    for v in pen_mins.values():
        x_C.append(v["C"])
        y_mse.append(v["mse"])

    ax.vlines(x=x_C, ymin=ax.get_ylim()[0], ymax=y_mse, ls="dashed", color="k", alpha=0.25)
    ax.scatter(x_C, y_mse, color="k", marker="x", alpha=0.25)
    ax.set(xlabel="C")
    ax.set_xscale("log")
    ax.legend(ncol=2)
    # plot_utils.save(filename + "C")
    plt.show()

if __name__ == "__main__":
    rnd = 3211
    X, y = utils.get_fifa_data(n=10000, random_state=rnd)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=3/4, random_state=rnd)


    LinearModel(X_train, y_train, filename="BiasVar_LinearRegression", random_state=rnd)
    Singel_tree_increasing_depth(X, y, filename="BiasVar_SingleTree", random_state=rnd)
    Trees_increasing_ensamble(X, y, filename="BiasVar_Bag_and_Rf.pdf")
    Boosting(X, y, filename="Boosting")
    SupperVecReg(X, y, filename="BiasVar_SVR", random_state=rnd)