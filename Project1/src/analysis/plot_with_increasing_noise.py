import context
from sknotlearn.datasets import plot_FrankeFunction
import numpy as np

v = np.linspace(0, 1, int(np.sqrt(600)))

sigma = [0, 0.01, 0.1, 0.2]

for s in sigma:
    s_str = str(s).replace(".", "_")
    filename = f"../../tex/figs/franke_functions_{s_str}"
    plot_FrankeFunction(v,v, s, filename=filename)