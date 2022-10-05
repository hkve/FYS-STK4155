from turtle import title
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.resampling import Bootstrap
from sknotlearn.data import Data
from utils import make_figs_path, colors, model_names
from ridgelasso import load 


#TODO: Make plots for various lambda (make one for optimal_lambda and one for a big one). Both for Ridge and LASSO. 
#TODO: Bias-Var with dependence on lambda for a given degree (8) (Ridge and LASSO)
#TODO: Make a common plot for train/test mse for the various models

""" 
The idea behind this document is to create functions for two things: exploring the effect of bootstrapping (various number of rounds), and implementing bootstrapping when analysing MSE (and bias and variance) as a funtion of model complexity. 
"""
fontsize_leg = 14
fontsize_lab = 14
fontsize_tit = 16

def plot_hist_of_bootstrap(Bootstrap_, degree, model=LinearRegression):
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
    plt.hist(mse_train, bins=find_bins(mse_train), label='MSE for training data', color=colors[1], density=True)
    plt.hist(mse_test, bins=find_bins(mse_test), alpha=0.6,  label='MSE for test data', color=colors[0], density=True)
    plt.xlabel('MSE', fontsize=fontsize_lab)
    plt.ylabel('Probability density', fontsize=fontsize_lab)
    plt.title(f'MSE when bootstrapping {Bootstrap_.rounds} times: {model_names[model]}', fontsize=fontsize_tit)
    plt.legend(fontsize = fontsize_leg)
    # plt.savefig(make_figs_path(f'BS_hist_bootstraped_{Bootstrap_.rounds}_rounds_of_degree_{degree}.pdf'), dpi=300)
    plt.show()


def plot_mse_across_rounds(Bootstrap_list_rounds, rounds, model=LinearRegression):
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
    plt.title(f'MSE across rounds: {model_names[model]}', fontsize=fontsize_tit)
    # plt.savefig(make_figs_path(f'BS_mse_across_rounds_{model_names[model]}.pdf'), dpi=300)
    plt.show()


def plot_train_test_mse(Bootstrap_list, degrees, model):
    """Plotting the training and test mse for the various polynomial degrees. 

    Args:
        Bootstrap_list (list): List consisting of instances of the class Bootstrap. Each element of the class responds to a certain polynomial degree and each element has been bootstrapped the same number of rounds.
        degrees (array): Array of the polynomial degrees 
    """
    #Extracting the mean value for each degree:
    mse_train = [BS.mse_train for BS in Bootstrap_list]
    mse_test = [BS.mse_test for BS in Bootstrap_list]

    #Extracting the models which sum to the mean: 
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
    plt.legend(fontsize = fontsize_leg)
    plt.title(f'Train and Test MSE: {model_names[model]}', fontsize=fontsize_tit)
    # plt.savefig(make_figs_path(f'BS_train_test_mse_{model_names[model]}.pdf'), dpi=300)
    plt.show()


def plot_bias_var(Bootstrap_list, degrees, model, **kwargs):
    """Plotting the bias-variance decomposition of the mse value across various polynomial degrees. 

    Args:
        Bootstrap_list (list): List consisting of instances of the class Bootstrap. Each element of the class responds to a certain polynomial degree and each element has been bootstrapped the same number of rounds.  
        degrees (array): Array of the polynomial degrees 
    """
    opt = {
        "title": f"Bias-Variance Decomposition of the MSE: {model_names[model]}",
        "filename": f"BS_Bias_var_decomp_{model_names[model]}.pdf"
    }
    opt.update(kwargs)

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
    plt.plot(degrees, proj_mse, '--', label=r'Bias$^2$ + Variance', c=colors[3], lw=3)
    plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
    plt.ylabel('Score', fontsize=fontsize_lab)
    plt.title(opt["title"], fontsize=fontsize_tit)
    plt.legend(fontsize = fontsize_leg)
    plt.savefig(make_figs_path(opt["filename"]), dpi=300)
    plt.show()


def solve_c(y, X, degree, rounds, reg, scale_scheme='Standard', ratio=3/4, random_state=321):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    data = Data(y, X_poly)
    data_train, data_test = data.train_test_split(ratio=ratio, random_state=random_state)
    data_train = data_train.scaled(scheme=scale_scheme)
    data_test = data_train.scale(data_test)

    BS = Bootstrap(reg, data_train, data_test, random_state = random_state, rounds=rounds)
    return BS


def bootstrap_across_rounds(model, rounds, degree, lmbda=None):
    """Function for doing the analysis of bootstrap across rounds given a certain model. Plots histogram (if OLS) and mse across rounds. 

    Args:
        model (_type_): Model for linear regression
        rounds (ndarray): Number of bootstrap rounds 
        degree (int): Polynomial degree of model
        lmbda (int, optional): Lambda implemented in the Ridge and Lasso model. Defaults to None. 
    """
    Bootstrap_list_rounds = []
    for round in rounds:
        if model in [Ridge, Lasso]:
            reg = model(lmbda = lmbda)
        else: 
            reg = model()

        BS = solve_c(y, X, degree=degree, rounds=round, reg=reg)
        Bootstrap_list_rounds.append(BS)

        if model == LinearRegression and round in [30,102,408,606]:
            pass
            # plot_hist_of_bootstrap(BS, degree, model)
    plot_mse_across_rounds(Bootstrap_list_rounds, rounds, model)


def bootstrap_across_degrees(model, round, degrees, lmbdas=None):
    """Function for doing the analysis of bootstrap across degrees given a certain model and number of bootstrap rounds. Used to calculate the necesities for plotting test/train mse and bias-variance-decomposition. 

    Args:
        model (_type_): Model for linear regression
        round (int): number of bootstrap rounds
        degrees (ndarray): degress on which to preform the analysis
        lmbdas (ndarray, optional): array with same size as degrees. Each element of lmbdas corresponds to a polynomial degree. For Ridge and Lasso.  Defaults to None.
    """
    #For various degrees
    Bootstrap_list = []
    for i, deg in enumerate(degrees): 
        if model in [Ridge, Lasso]:
            reg = model(lmbda = lmbdas[i])
        else: 
            reg = model()

        BS = solve_c(y, X, degree=deg, rounds=round, reg=reg)
        Bootstrap_list.append(BS)

    # plot_train_test_mse(Bootstrap_list, degrees, model)
    # plot_bias_var(Bootstrap_list, degrees, model)

    return Bootstrap_list


# def plot_comparison(BS_lists, models=[LinearRegression, Ridge, Lasso]):
#     #Plotting the train/test for various models
#     sns.set_style('darkgrid')
#     for i, Bootstrap_list in enumerate(BS_lists):
#         #Extracting the mean value for each degree:
#         mse_train = [BS.mse_train for BS in Bootstrap_list]
#         mse_test = [BS.mse_test for BS in Bootstrap_list]

#         #Extracting the models which sum to the mean 
#         # mse_trains = np.array([BS.mse_train_values for BS in Bootstrap_list])
#         # mse_tests = np.array([BS.mse_test_values for BS in Bootstrap_list])

#         #Plotting: 

#         # plt.plot(degrees, mse_trains, c=colors[0], lw=.5, alpha=0.2)
#         # plt.plot(degrees, mse_tests, c=colors[1], lw=.5, alpha=0.2)

#         plt.plot(degrees, mse_train, label='Train MSE', c=colors[i], lw=2.5, alpha=0.75)
#         plt.plot(degrees, mse_test, label='Test MSE', c=colors[i], lw=2.5)

#     plt.xlim([1,15])
#     plt.ylim([0,0.6])
#     plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
#     plt.ylabel('MSE value', fontsize=fontsize_lab)
#     plt.legend(fontsize = fontsize_leg)
#     plt.title(f'Train and Test MSE: All Models', fontsize=fontsize_tit)
#     plt.savefig(make_figs_path(f'BS_train_test_mse_all_models.pdf'), dpi=300)
#     plt.show()


if __name__ == "__main__":
    D = make_FrankeFunction(n=600, uniform=True, random_state=321, noise_std=0.1)
    y, X = D.unpacked()

    rounds = np.arange(30, 1000+1, (1001-30)//100)
    degrees = np.arange(1, 15+1)
    round = 400

###
#Linreg
###

    # bootstrap_across_rounds(LinearRegression, rounds, degree=7) #Chose deg=7 rather 'randomly' 
    # BS_list_OLS = bootstrap_across_degrees(LinearRegression, round, degrees)
    # plot_train_test_mse(BS_list_OLS, degrees, LinearRegression)
    # plot_bias_var(BS_list_OLS, degrees, LinearRegression)


###
#Ridge
###
    #Find the best lambda and degree:
    degrees_grid, lmbdas_grid, MSEs = load("ridge_grid")
    
    optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
    optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
    optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]

    # bootstrap_across_rounds(Ridge, rounds, degree=optimal_degree, lmbda=optimal_lmbda) #Shows that round=400 gives the stabilized state
    # BS_list_R = bootstrap_across_degrees(Ridge, round, degrees, lmbdas=optimal_lmbdas)
    # plot_train_test_mse(BS_list_R, degrees, Ridge)
    # plot_bias_var(BS_list_R, degrees, Ridge)

###
#LASSO
###

    #Find the best lambda and degree:
    degrees_grid, lmbdas_grid, MSEs = load("lasso_grid")
    
    optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
    optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
    optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]

    # bootstrap_across_rounds(Lasso, rounds, degree=optimal_degree, lmbda=optimal_lmbda) #Shows that round=400 gives the stabilized state
    # BS_list_L = bootstrap_across_degrees(Lasso, round, degrees, lmbdas=optimal_lmbdas)
    # plot_train_test_mse(BS_list_L, degrees, Lasso)
    # plot_bias_var(BS_list_L, degrees, Lasso)

###
#For various n's
###
#Wish to plot the train/test and bias-var decomposition for various n's. Should  be in the same plot?
    ns = [60, 600, 1500]
    for n in ns:
        D = make_FrankeFunction(n=n, uniform=True, random_state=321, noise_std=0.1)
        y, X = D.unpacked()

        BS_list = bootstrap_across_degrees(LinearRegression, round, degrees)
        plot_bias_var(BS_list, degrees, LinearRegression, title=f"Bias-Variance Decomposition: OLS, {n} data points", filename=f"BS_Bias_var_decomp_OLS_{n}_data_points.pdf")








###
#Comparison
###


    

    

    


