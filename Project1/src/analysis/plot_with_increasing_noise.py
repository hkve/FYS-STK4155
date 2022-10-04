import numpy as np

import context
from sknotlearn.data import Data
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction, load_Terrain, plot_Terrain
from sknotlearn.linear_model import LinearRegression
from utils import make_figs_path


def plot_with_increasing_noise(sigma = [0, 0.01, 0.1, 0.2]):
    for s in sigma:
        D = make_FrankeFunction(625, uniform=False, noise_std=s, random_state=321)
        s_str = str(s).replace(".", "_")
        filename = make_figs_path(f"franke_functions_{s_str}")
        plot_FrankeFunction(D, filename=filename)

def plot_Terrain_fit_and_raw():
    D = load_Terrain("SRTM_data_Nica.tif")

    plot_Terrain(D)

    Dp = D.polynomial(degree=10)
    Dp_train, Dp_test = Dp.train_test_split(ratio=2/3, random_state=321)

    Dp_train = Dp_train.scaled(scheme="Standard")
    Dp_test = Dp_train.scale(Dp_test)

    reg = LinearRegression().fit(Dp_train)

    Dp_test_predict = Data(reg.predict(Dp_train.X), Dp_train.X)
    Dp_test_predict = Dp_train.unscale(Dp_test_predict)

    Dp_test = Dp_train.unscale(Dp_test)
    plot_Terrain(Dp_test)
    plot_Terrain(Dp_test_predict)

if __name__ == "__main__":
    # plot_with_increasing_noise()

    plot_Terrain_fit_and_raw()