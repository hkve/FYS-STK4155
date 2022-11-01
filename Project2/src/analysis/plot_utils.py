import matplotlib.pyplot as plt
from cycler import cycler
import seaborn as sns

colors = [
    sns.color_palette('husl')[-3],
    sns.color_palette('husl')[-2],
    sns.color_palette('husl')[-1],
    'mediumorchid'
]

sns.set_style("darkgrid")
# Set all fonts to be equal to tex
# https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["text.usetex"] = True

# Saving parameters
plt.rcParams["savefig.dpi"] = 300

# Figure options, set tight layout
plt.rc("figure", autolayout=True)

# Font sizes
plt.rc("axes", titlesize=18, labelsize=16, prop_cycle=cycler('color', colors))
plt.rc("legend", fontsize=14, shadow=True)

# Tick parameters
_ticks_default_parameters = {
    "labelsize": 12
}
plt.rc("xtick", **_ticks_default_parameters)
plt.rc("ytick", **_ticks_default_parameters)

# Line options
plt.rc("lines", linewidth=2)


# To see more paramteres, print the possible options:
# print(plt.rcParams)
