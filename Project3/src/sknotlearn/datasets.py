import os
import sys
from imageio import imread
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis.plot_utils as plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pathlib as pl
from fifa21_EAsports_reader import load_fifa

def make_FrankeFunction(n=1000, linspace=False, noise_std=0, random_state=42):
    x, y = None, None

    np.random.seed(random_state)
    if linspace:
        perfect_square = (int(n) == int(np.sqrt(n))**2)
        assert perfect_square, f"{n = } is not a perfect square. Thus linspaced points cannot be made"

        x = np.linspace(0, 1, int(np.sqrt(n)))
        y = np.linspace(0, 1, int(np.sqrt(n)))

        X, Y = np.meshgrid(x, y)
        x = X.flatten()
        y = Y.flatten()
    else:
        x = np.random.uniform(low=0, high=1, size=n)
        y = np.random.uniform(low=0, high=1, size=n)

    z = FrankeFunction(x, y) + np.random.normal(loc=0, scale=noise_std, size=n)

    return Data(z, np.c_[x,y])


def make_debugdata(n=100, scale=0.1, random_state=321):
    np.random.seed(random_state)
    x = np.random.uniform(-1, 1, n)
    y = x**2 + np.random.normal(scale=scale, size=n)
    X = np.c_[np.ones(n), x, x**2]

    return x, y, X


def plot_surf(D):
    sns.set_style("white")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    X = D.X
    if D.X.shape[1] == 2:
        surf = ax.plot_trisurf(*X.T, D.y, cmap=plot_utils.cmap_terrain, linewidth=0, antialiased=False)
    else:
        surf = ax.plot_trisurf(X[:,1], X[:,2], D.y, cmap=plot_utils.cmap_terrain, linewidth=0, antialiased=False)

    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig, ax, surf, cbar


def plot_FrankeFunction(D, angle=(18, 45), filename=None):
    fig, ax, surf, cbar = plot_surf(D)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_zlabel(r"$F (x,y)$", fontsize=14, rotation=90)
    ax.view_init(*angle)

    if filename:
        plt.savefig(filename, dpi=300)

    fig.tight_layout()
    plt.show()


def plot_Terrain(D, angle=(18,45), figsize=(10,7), filename=None):
    fig, ax, surf, cbar = plot_surf(D)

    fig.set_size_inches(*figsize)
    ax.set_zlabel(r"Terrain", fontsize=14, rotation=90)
    ax.view_init(*angle)

    if filename:
        plt.savefig(filename, dpi=300)

    fig.tight_layout()
    plt.show()


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def load_Terrain(filename="SRTM_data_Nica.tif", n=900, random_state=321):
    path = pl.Path(__file__).parent / filename
    start, stop = 1600, 1900

    assert n <= (stop-start)**2, f"Cannot load {n} points of terrain data, maximum available is {(stop-start)**2}."

    z = imread(path)[start:stop, start:stop]

    # drawing random samples from grid
    np.random.seed(random_state)
    x1 = np.arange(stop=stop-start) # NS-coordinates
    x2 = x1.copy() # EW-coordinates
    # Making array of every combination of (x1, x2)
    X = np.reshape(np.meshgrid(x1, x2), (2, (stop-start)**2)).T
    np.random.shuffle(X) # shuffling for randomness
    X = X[:n] # drawing n points

    y = np.zeros(shape=n, dtype=float)
    for i, (x1,x2) in enumerate(X):
        y[i] = z[x1,x2]

    return Data(y, X.astype(float))


def load_BreastCancer(filename="breast_cancer.csv"):
    import pandas as pd

    path = pl.Path(__file__).parent / filename
    df = pd.read_csv(path)

    # Target (y) is saved under "diagnosis" col with values M = malignant (bad), B = benign (good)
    # Save these as M == 1, B == 0
    y = np.where(df.diagnosis == "M", 1, 0).astype(int)

    # Drop the target col and id
    df.drop(columns=["diagnosis", "id"], inplace=True)

    # One strange col named "Unnamed: 32" with all NaNs want to stay for some reason...
    # Drop cols where all values as NaN
    df.dropna(how="all", axis=1, inplace=True)
    X = df.to_numpy()

    # Save col names
    col_names = list(df.columns)

    return Data(y, X, col_names=col_names)


def load_MNIST():
    from tensorflow.keras.utils import get_file

    DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

    path = get_file('mnist.npz', DATA_URL)
    with np.load(path) as data:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        x_train = data['x_train']
        y_train = np.array([np.where(y == labels, 1, 0)
                            for y in data['y_train']])

        x_test = data['x_test']
        y_test = np.array([np.where(y == labels, 1, 0)
                           for y in data['y_test']])

    return x_train, y_train, x_test, y_test


def load_CIFAR10():
    from tensorflow.keras import datasets

    labels = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    (x_train, train_labels), (x_test, test_labels) = datasets.cifar10.load_data()

    y_train = np.array([np.where(y == range(len(labels)), 1, 0) for y in train_labels])
    y_test = np.array([np.where(y == range(len(labels)), 1, 0) for y in test_labels])

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    D = load_CIFAR10()
