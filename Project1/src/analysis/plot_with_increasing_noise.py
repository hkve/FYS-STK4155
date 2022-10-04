import context
from sknotlearn.data import Data
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction, load_Terrain, plot_Terrain
from utils import make_figs_path
import numpy as np


def plot_with_increasing_noise(sigma = [0, 0.01, 0.1, 0.2]):
    for s in sigma:
        D = make_FrankeFunction(625, uniform=False, noise_std=s, random_state=321)
        s_str = str(s).replace(".", "_")
        filename = make_figs_path(f"franke_functions_{s_str}")
        plot_FrankeFunction(D, filename=filename)

if __name__ == "__main__":
    # plot_with_increasing_noise()

    D = load_Terrain("SRTM_data_Nica.tif")
    plot_Terrain(D, angle=(30,148), filename=make_figs_path("Nica_terrain_plot"))
