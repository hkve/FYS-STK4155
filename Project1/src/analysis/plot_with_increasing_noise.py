import context
from sknotlearn.data import Data
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction
import numpy as np

sigma = [0, 0.01, 0.1, 0.2]

for s in sigma:
    D = make_FrankeFunction(625, uniform=False, noise_std=s, random_state=321)
    s_str = str(s).replace(".", "_")
    filename = f"../../tex/figs/franke_functions_{s_str}.pdf"
    plot_FrankeFunction(D, filename=filename)