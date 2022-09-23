import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def make_FrankeFunction(n=1000, uniform=True, noise_std=0, random_state=42):
	x, y = None, None

	np.random.seed(random_state)
	if uniform:
		x = np.random.uniform(low=0, high=1, size=n)
		y = np.random.uniform(low=0, high=1, size=n)
	else:
		x = np.linspace(0, 1, n)
		y = np.linspace(0, 1, n)

	z = FrankeFunction(x, y) + np.random.normal(loc=0, scale=noise_std, size=n)

	return np.c_[x, y], z

def plot_FrankeFunction(x, y, noise_std=0, filename=None):
	X, Y = np.meshgrid(x, y)
	Z = FrankeFunction(X, Y) + np.random.normal(loc=0, scale=noise_std, size=(len(x), len(y)))

	fig = plt.figure()
	sns.set_style("white")
	ax = fig.gca(projection="3d")
	surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)

	ax.set_zlim(-0.10, 1.40)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

	ax.set_xlabel("x", fontsize=14)
	ax.set_ylabel("y", fontsize=14)
	ax.set_zlabel(r"$F (x,y)$", fontsize=14, rotation=90)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	
	polar_angle = 18 # This is "theta" is physics convention of spherical coordinates
	azimuthal_angle = 45 # "phi"
	ax.view_init(polar_angle, azimuthal_angle)
	
	if filename:
		plt.savefig(filename, dpi=300)

	plt.show()


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

if __name__ == "__main__":
	X, y = make_FrankeFunction(n=100)
	plot_FrankeFunction(*X.T)