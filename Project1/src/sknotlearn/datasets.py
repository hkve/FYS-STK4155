import os
import sys
from imageio import imread
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Data 

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
		perfect_square = ( int(n) == int(np.sqrt(n))**2)
		assert perfect_square, f"{n = } is not a perfect square. Thus linspaced points cannot be made"

		x = np.linspace(0, 1, int(np.sqrt(n)))
		y = np.linspace(0, 1, int(np.sqrt(n)))

		X, Y = np.meshgrid(x, y)
		x = X.flatten()
		y = Y.flatten()
		

	z = FrankeFunction(x, y) + np.random.normal(loc=0, scale=noise_std, size=n)

	return Data(z, np.c_[x,y])

def plot_FrankeFunction(D, angle=(18, 45), filename=None, Franke=True):
	sns.set_style("white")
	fig = plt.figure()
	ax = fig.add_subplot(projection="3d")
	surf = ax.plot_trisurf(*D.X.T, D.y, cmap=cm.viridis, linewidth=0, antialiased=False)

	#TODO: Add limits
	if Franke:
		ax.set_zlim(-0.10, 1.40)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

	ax.set_xlabel("x", fontsize=14)
	ax.set_ylabel("y", fontsize=14)
	ax.set_zlabel(r"$F (x,y)$", fontsize=14, rotation=90)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	
	ax.view_init(*angle)
	
	plt.tight_layout()

	if filename:
		plt.savefig(filename, dpi=300)
	
	plt.show()

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


def load_terrain(filename="SRTM_data_mot.tif"):
	l = np.arange(30)
	X, Y = np.meshgrid(l,l)	
	x = X.flatten()
	y = Y.flatten()

	z = imread(filename)[1740:1800:2, 1600:1660:2] 
	z = z.flatten()
	
	return Data(z, np.c_[x,y])

def plot_terrain(D):

	fig = plt.figure()
	ax = fig.gca(projection="3d")

	im = ax.plot_surface(X, Y, D, cmap="viridis")
	fig.colorbar(im)
	plt.show()


if __name__ == "__main__":
	D = make_FrankeFunction(n=625, uniform=False, noise_std=0.1)
	# plot_FrankeFunction(D)

	D = load_terrain()
	plot_FrankeFunction(D, angle=(22,-55), Franke=False)
