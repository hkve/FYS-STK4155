import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import time

import context
import sknotlearn.optimize as opt
from sknotlearn.data import Data
from sknotlearn.datasets import load_Terrain, plot_Terrain


def optimize_OLS(y, X):
    return np.linalg.pinv(X.T @ X) @ X.T @ y


def fit_terrain(data_train: Data, data_val: Data, filename: str = None):
    theta_opt = optimize_OLS(*data_train.unpacked())
    data_pred = Data(data_val.X @ theta_opt, data_val.X)
    return data_pred


if __name__ == "__main__":
    D = load_Terrain(random_state=123, n=600).polynomial(degree=11)
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=42)
    D_train = D_train.scaled(scheme="Standard")
    D_test = D_train.scale(D_test)

    D_pred = fit_terrain(D_train, D_test)

    print(f"OLS MSE: {np.mean((D_pred.y - D_test.y)**2)}")

    plot_Terrain(D_train.unscale(D_test), angle=(16, -165))
    plot_Terrain(
        D_train.unscale(D_pred),
        angle=(16, -165),
        filename=plot_utils.make_figs_path("terrain_OLS")
    )
