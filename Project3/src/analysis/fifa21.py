import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import plot_utils
import fifa21_utils as utils

import matplotlib.ticker as mticker

def LinearModel_comparison(X, y, random_state=321):
    alpha = np.linspace(0.01, 4, 100)
    
    ls = {
        "mse": "solid",
        "bias": "dashed",
        "var": "dotted"
    }

    c = {
        Ridge: plot_utils.colors[0],
        Lasso: plot_utils.colors[1],
    }

    method_params = {Ridge: None, Lasso: {"max_iter": 10000}}
    
    fig, ax = plt.subplots()
    for Reg in [Ridge, Lasso]:
        mse, bias, var = utils.bootstrap(X_train, y_train, Reg, "alpha", alpha, method_params=method_params[Reg], random_state=rnd)
        
        # ax.plot([], [], " ", label=Reg.__name__)
        ax.plot(alpha, mse, label="mse", ls=ls["mse"], c=c[Reg])
        ax.plot(alpha, bias, label="bias", ls=ls["bias"], c=c[Reg])
        ax.plot(alpha, var, label="var", ls=ls["var"], c=c[Reg])

    ax.set_yscale("log")
    old_lim = ax.get_ylim()
    print(old_lim)
    new_lim = (old_lim[0], old_lim[1]*1.4)
    print(new_lim)
    ax.legend(ncol=2)
    ax.set(xlabel=r"Regularisation $\alpha$", ylim=new_lim)
    plt.show()



if __name__ == "__main__":
    rnd = 3211
    X, y = utils.get_fifa_data(n=10000, random_state=rnd)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=3/4, random_state=rnd)

    LinearModel_comparison(X_train, y_train, random_state=rnd)