# This script serves to plot data for visualisation.
import context
from sknotlearn.data import Data
from sknotlearn.datasets import make_FrankeFunction, plot_FrankeFunction, load_Terrain, plot_Terrain
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from utils import make_figs_path


def plot_with_increasing_noise(n=625, sigma:list = [0, 0.01, 0.1, 0.2], random_state=321):
    """Plots the Franke function with different noise magnitudes.

    Args:
        n (int, optional): Number of data points, must be a square number. Defaults to 625.
        sigma (list, optional): Noise stds to us when generating different data to plot. Defaults to [0, 0.01, 0.1, 0.2].
        random_state (int, optional): np.random seed. Defaults to 321.
    """
    for s in sigma:
        D = make_FrankeFunction(n=n, uniform=False, noise_std=s, random_state=random_state)
        s_str = str(s).replace(".", "_")
        filename = make_figs_path(f"franke_functions_{s_str}")
        plot_FrankeFunction(D, filename=filename)

def plot_Terrain_fit_and_raw(n:int=600, degree:int=12, model:Data=LinearRegression(), angle=(16,-165), random_state:int=321):
    """Plots the full terrain with n points, the test portion of the terrain, and the prediciton from the model

    Args:
        n (int, optional): Number of data points. Defaults to 600.
        degree (int, optional): Polynomial degree of design matrix. Defaults to 12.
        model (Data, optional): Model to use when predicting test data. Defaults to LinearRegression().
        angle (tuple, optional): Polar, azimuthal angle to view 3D plot from. Defaults to (16,-165).
        random_state (int, optional): np.radnom seed. Defaults to 321.
    """
    D = load_Terrain("SRTM_data_Nica.tif", n=n)

    plot_Terrain(D)

    Dp = D.polynomial(degree=degree)
    Dp_train, Dp_test = Dp.train_test_split(ratio=2/3, random_state=random_state)

    Dp_train = Dp_train.scaled(scheme="Standard")
    Dp_test = Dp_train.scale(Dp_test)

    reg = model.fit(Dp_train)

    Dp_test_predict = Data(reg.predict(Dp_train.X), Dp_train.X)
    Dp_test_predict = Dp_train.unscale(Dp_test_predict)

    Dp_test = Dp_train.unscale(Dp_test)
    plot_Terrain(Dp_test, angle=angle)
    plot_Terrain(Dp_test_predict, angle=angle)

if __name__ == "__main__":
    plot_with_increasing_noise()

    plot_Terrain_fit_and_raw()