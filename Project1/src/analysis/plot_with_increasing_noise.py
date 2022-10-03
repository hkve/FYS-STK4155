import context
from sknotlearn.data import Data
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction, load_terrain
from utils import make_figs_path
import numpy as np

sigma = [0, 0.01, 0.1, 0.2]

D = load_terrain()
plot_FrankeFunction(D, angle=(22,-55), filename=make_figs_path('terrain_plot'))
exit()
for s in sigma:
    D = make_FrankeFunction(625, uniform=False, noise_std=s, random_state=321)
    s_str = str(s).replace(".", "_")
    filename = f"../../tex/figs/franke_functions_{s_str}.pdf"
    plot_FrankeFunction(D, filename=filename)