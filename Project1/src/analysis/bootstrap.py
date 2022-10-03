from code import interact
from statistics import variance
import numpy as np
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.resampling import Bootstrap
from sknotlearn.data import Data
from utils import make_figs_path, colors
from ridgelasso import load 

""" 
With the bias and var calculated across the bootstrap
(With data-class implemented)
"""
fontsize_lab = 14
fontsize_tit = 16

def plot_hist_of_bootstrap(Bootstrap_, degree):
    """Generate the histogram of the mse-values of a certain model with a given number 
    of bootstrap rounds

    Args:
        Bootstrap_ (Bootstrap): bootstrap of a certain degree
        degree (int): Degree of the model which is bootstrapped
    """
    mse_train = Bootstrap_.mse_train_values
    mse_test = Bootstrap_.mse_test_values

    find_bins = lambda arr, times=800: np.abs(int((np.max(arr)-np.min(arr))*times))

    sns.set_style('darkgrid')
    plt.hist(mse_train, bins=find_bins(mse_train), label='mse for training data', color=colors[1], density=True)
    plt.hist(mse_test, bins=find_bins(mse_test), alpha=0.6,  label='mse for test data', color=colors[0], density=True)
    plt.xlabel('MSE', fontsize=fontsize_lab)
    plt.ylabel('Probability density', fontsize=fontsize_lab)
    plt.title(f'Results of MSE when bootstrapping {Bootstrap_.rounds} times', fontsize=fontsize_tit)
    plt.legend()
    # plt.savefig(make_figs_path(f'hist_bootstrap_{Bootstrap_.rounds}_scaled_of_degree_{degree}.pdf'), dpi=300)
    plt.show()


def plot_bias_var(Bootstrap_list, degrees):
    """Plotting the bias-variance decomposition of the mse value across various polynomial degrees. 

    Args:
        Bootstrap_list (list): List consisting of instances of the class Bootstrap. Each element of the class responds to a certain polynomial degree and each element has been bootstrapped the same number of rounds.  
        degrees (array): Array of the polynomial degrees 
    """
    #extracting the mean values for each degree: 
    mse = [BS.mse_test for BS in Bootstrap_list]
    bias = [BS.bias_test for BS in Bootstrap_list]
    var = [BS.var_test for BS in Bootstrap_list]
    proj_mse = [BS.projected_mse for BS in Bootstrap_list]
    
    #Plotting:
    sns.set_style('darkgrid')
    plt.plot(degrees, mse, label='MSE', c=colors[2], lw=2.5)
    plt.plot(degrees, bias, label=r'Bias$^2$', c=colors[1], lw=2.5)
    plt.plot(degrees, var, label='Variance', c=colors[0], lw=2.5)
    plt.plot(degrees, proj_mse, '--', label='Projected mse', c=colors[3], lw=3)
    plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
    plt.ylabel('Score', fontsize=fontsize_lab)
    plt.title('Bias-Variance Decomposition of the MSE', fontsize=fontsize_tit)
    plt.legend()
    # plt.savefig(make_figs_path(f'Bias_var_decomp.pdf'), dpi=300)
    plt.show()


def plot_mse(Bootstrap_list, degrees):
    """Plotting the training and test mse for the various polynomial degrees. 

    Args:
        Bootstrap_list (list): List consisting of instances of the class Bootstrap. Each element of the class responds to a certain polynomial degree and each element has been bootstrapped the same number of rounds.
        degrees (array): Array of the polynomial degrees 
    """
    #Extracting the mean value for each degree:
    mse_train = [BS.mse_train for BS in Bootstrap_list]
    mse_test = [BS.mse_test for BS in Bootstrap_list]

    #Extracting the models which sum to the mean 
    mse_trains = np.array([BS.mse_train_values for BS in Bootstrap_list])
    mse_tests = np.array([BS.mse_test_values for BS in Bootstrap_list])

    #Plotting: 
    sns.set_style('darkgrid')
    plt.plot(degrees, mse_trains, c=colors[0], lw=.5, alpha=0.2)
    plt.plot(degrees, mse_tests, c=colors[1], lw=.5, alpha=0.2)

    plt.plot(degrees, mse_train, label='Train MSE', c=colors[0], lw=2.5, alpha=0.75)
    plt.plot(degrees, mse_test, label='Test MSE', c=colors[1], lw=2.5)

    plt.ylim([0,0.6])
    plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
    plt.ylabel('MSE value', fontsize=fontsize_lab)
    plt.legend()
    plt.title('Training and Test MSE', fontsize=fontsize_tit)
    # plt.savefig(make_figs_path(f'train_test_mse.pdf'), dpi=300)
    plt.show()


def plot_mse_across_rounds(Bootstrap_list_rounds, rounds):
    """Plotting the mean of the MSE as a function of rounds

    Args:
        Bootstrap_list_rounds (list): A list of Bootstrap instances 
        rounds (int): Number of bootstrap-rounds 
    """
    mse = [BS.mse_test for BS in Bootstrap_list_rounds]
    mse_std = [np.std(BS.mse_test_values) for BS in Bootstrap_list_rounds]/np.sqrt(rounds)
    sns.set_style('darkgrid')
    plt.plot(rounds, mse, c=colors[0], lw=2.5)
    plt.fill_between(rounds, mse+mse_std, mse-mse_std, color=colors[1], alpha=.3)
    plt.xlabel('Bootstrap rounds', fontsize=fontsize_lab)
    plt.ylabel('MSE value', fontsize=fontsize_lab)
    plt.title('MSE across rounds')
    # plt.savefig(make_figs_path(f'mse_across_rounds1.pdf'), dpi=300)
    plt.show()


def solve_c(y, X, degree, rounds, reg, scale_scheme='Standard', ratio=3/4, random_state=321 ):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    data = Data(y, X_poly)
    data_train, data_test = data.train_test_split(ratio=ratio, random_state=random_state)
    data_train = data_train.scaled(scheme=scale_scheme)
    data_test = data_train.scale(data_test)

    BS = Bootstrap(reg, data_train, data_test, random_state = random_state, rounds=rounds)
    return BS

# def run_bootstrap(model, lmbda=None):
#     #For various rounds:
#     rounds = np.arange(30, 1000+1, (1001-30)//10)
#     Bootstrap_list_rounds = []
#     for i, round in enumerate(rounds):
#         if model in [Ridge, Lasso]:
#             reg = model(lmbda = lmbda)
#         else: 
#             reg = model()
#         BS = solve_c(y, X, degree=7, rounds=round, reg=reg)
#         Bootstrap_list_rounds.append(BS)
#         if model == LinearRegression:
#             pass 
#             # plot_hist_of_bootstrap(BS, 7)
#     plot_mse_across_rounds(Bootstrap_list_rounds, rounds)

#     #For various degrees
#     degrees = np.arange(1, 12+1)
#     Bootstrap_list = []
#     for deg in degrees: 
#         BS = solve_c(y, X, degree=deg, rounds=70, reg=reg)
#         Bootstrap_list.append(BS)

#     plot_mse(Bootstrap_list, degrees)
#     plot_bias_var(Bootstrap_list, degrees)


if __name__ == "__main__":
    D = make_FrankeFunction(n=600, uniform=True, random_state=321, noise_std=0.1)
    y, X = D.unpacked()


###
#Linreg
###

    #For various rounds:
    rounds = np.array([30, 120, 210, 300])
    rounds = np.arange(30, 1000+1, (1001-30)//10)
    Bootstrap_list_rounds = []
    for i, round in enumerate(rounds):
        BS = solve_c(y, X, degree=7, rounds=round, reg=LinearRegression())
        Bootstrap_list_rounds.append(BS)
        # plot_hist_of_bootstrap(BS, 7)
    plot_mse_across_rounds(Bootstrap_list_rounds, rounds)

    #For various degrees
    degrees = np.arange(1, 12+1)
    Bootstrap_list = []
    for deg in degrees: 
        BS = solve_c(y, X, degree=deg, rounds=70, reg=LinearRegression())
        Bootstrap_list.append(BS)

    plot_mse(Bootstrap_list, degrees)
    plot_bias_var(Bootstrap_list, degrees)
    
    exit()

###
#Ridge
###
    #Find the best lambda and degree:
    degrees_grid, lmbdas_grid, MSEs = load("ridge_grid2")
    
    optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
    optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
    optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]

    #For various rounds:
    rounds = np.array([30, 120, 210, 300])
    rounds = np.arange(30, 1000+1, (1001-30)//10)
    Bootstrap_list_rounds = []
    for i, round in enumerate(rounds):
        BS = solve_c(y, X, degree=optimal_degree, rounds=round, reg=Ridge(lmbda = optimal_lmbda))
        Bootstrap_list_rounds.append(BS)
        # plot_hist_of_bootstrap(BS, 7)
    plot_mse_across_rounds(Bootstrap_list_rounds, rounds)

    #For various degrees
    degrees = np.arange(1, 12+1)
    lmbdas = []
    Bootstrap_list = []
    for i, deg in enumerate(degrees): 
        BS = solve_c(y, X, degree=deg, rounds=70, reg=Ridge(lmbda = optimal_lmbdas[i]))
        Bootstrap_list.append(BS)

    plot_mse(Bootstrap_list, degrees)
    plot_bias_var(Bootstrap_list, degrees)



###
#LASSO
###

    #Find the best lambda and degree:
    degrees_grid, lmbdas_grid, MSEs = load("lasso_grid")
    
    optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
    optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
    optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]

    #For various rounds:
    rounds = np.array([30, 120, 210, 300])
    rounds = np.arange(30, 1000+1, (1001-30)//10)
    Bootstrap_list_rounds = []
    for i, round in enumerate(rounds):
        BS = solve_c(y, X, degree=optimal_degree, rounds=round, reg=Ridge(lmbda = optimal_lmbda))
        Bootstrap_list_rounds.append(BS)
        # plot_hist_of_bootstrap(BS, 7)
    plot_mse_across_rounds(Bootstrap_list_rounds, rounds)

    #For various degrees
    degrees = np.arange(1, 12+1)
    lmbdas = []
    Bootstrap_list = []
    for i, deg in enumerate(degrees): 
        BS = solve_c(y, X, degree=deg, rounds=70, reg=Ridge(lmbda = optimal_lmbdas[i]))
        Bootstrap_list.append(BS)

    plot_mse(Bootstrap_list, degrees)
    plot_bias_var(Bootstrap_list, degrees)
    

    

    


