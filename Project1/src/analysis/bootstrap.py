from turtle import title
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction, load_Terrain
from sknotlearn.resampling import Bootstrap
from sknotlearn.data import Data
from utils import make_figs_path, colors, model_names
from ridgelasso import load 


#TODO: Bias-Var with dependence on lambda for a given degree (8) (LASSO)
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

    plt.ylim([0,mse_test[0]*2])
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
    # plt.savefig(make_figs_path(f"BS_Bias_var_lambdas_{model_names[model]}.pdf"), dpi=300)
    plt.show()


def plot_bias_var_lmbdas(Bootstrap_list, lmbdas, model):
    #extracting the mean values for each degree: 
    mse = [BS.mse_test for BS in Bootstrap_list]
    bias = [BS.bias_test for BS in Bootstrap_list]
    var = [BS.var_test for BS in Bootstrap_list]
    proj_mse = [BS.projected_mse for BS in Bootstrap_list]

    #Plotting:
    sns.set_style('darkgrid')
    plt.plot(lmbdas, mse, label='MSE', c=colors[2], lw=2.5)
    plt.plot(lmbdas, bias, label=r'Bias$^2$', c=colors[1], lw=2.5)
    plt.plot(lmbdas, var, label='Variance', c=colors[0], lw=2.5)
    plt.plot(lmbdas, proj_mse, '--', label=r'Bias$^2$ + Variance', c=colors[3], lw=3)
    plt.xscale('log')
    plt.xlabel(f'$\lambda$', fontsize=fontsize_lab)
    plt.ylabel('Score', fontsize=fontsize_lab)
    plt.title(f"Bias-Variance Decomposition of the MSE across $\lambda$: {model_names[model]}", fontsize=fontsize_tit)
    plt.legend(fontsize = fontsize_leg)
    plt.savefig(make_figs_path(f'BS_bias_var_lmbdas_{model_names[model]}.pdf'), dpi=300)
    plt.show()


def plot_bias_var_2lmbda(BS_list1, BS_list2, lmbda1, lmbda2, degrees, model):
    lmbdas = [lmbda1[0], lmbda2[0]]
    #Plotting:
    plt.figure(figsize=(8,8))
    sns.set_style('darkgrid')    
    for i, BS_list in enumerate([BS_list1, BS_list2]):
        #extracting the mean values for each degree: 
        mse = [BS.mse_test for BS in BS_list]
        bias = [BS.bias_test for BS in BS_list]
        var = [BS.var_test for BS in BS_list]

        if i==0:
            opt_string = " (opt)"
        else:
            opt_string = ""

        plt.plot(degrees, mse, label=f'MSE $\lambda = ${lmbdas[i]:.0E}'+opt_string, c=colors[i-1], lw=2.5)
        plt.plot(degrees, bias, label=f'Bias$^2$ $\lambda = ${lmbdas[i]:.0E}'+opt_string, c=colors[i-1], lw=2.5, ls='dotted')
        plt.plot(degrees, var, label=f'Variance $\lambda = ${lmbdas[i]:.0E}'+opt_string, c=colors[i-1], lw=2.5, ls='dashed')

    plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
    plt.ylabel('Score', fontsize=fontsize_lab)
    plt.title(f'Bias-Variance Decomposition for two $\lambda$: {model_names[model]}', fontsize=fontsize_tit)
    plt.legend(fontsize = 10)
    # plt.savefig(make_figs_path(f'BS_bias_var_two_lmbdas_{model_names[model]}.pdf'), dpi=300)
    plt.show()



def call_Bootstrap(y, X, degree, rounds, reg, scale_scheme='Standard', ratio=3/4, random_state=321):
    """A function which creates the design matrix, retrieves the scaled data, and calls Bootstrap. It returns an instance of Bootstrap. 

    Args:
        y (_type_): _description_
        X (_type_): _description_
        degree (_type_): _description_
        rounds (_type_): _description_
        reg (_type_): _description_
        scale_scheme (str, optional): _description_. Defaults to 'Standard'.
        ratio (_type_, optional): _description_. Defaults to 3/4.
        random_state (int, optional): _description_. Defaults to 321.

    Returns:
        _type_: _description_
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    data = Data(y, X_poly)
    data_train, data_test = data.train_test_split(ratio=ratio, random_state=random_state)
    data_train = data_train.scaled(scheme=scale_scheme)
    data_test = data_train.scale(data_test)

    BS = Bootstrap(reg, data_train, data_test, random_state=random_state, rounds=rounds)
    return BS


def bootstrap_across_rounds(D, model, rounds, degree, lmbda=None, hist=False):
    """Function for doing the analysis of bootstrap across rounds given a certain model. Plots histogram (if OLS) and mse across rounds. 

    Args:
        model (_type_): Model for linear regression
        rounds (ndarray): Number of bootstrap rounds 
        degree (int): Polynomial degree of model
        lmbda (int, optional): Lambda implemented in the Ridge and Lasso model. Defaults to None. 

    Returns: 
        Bootstrap_list_rounds (list): list of Bootstrap instances.
    """
    y, X = D.unpacked()

    Bootstrap_list_rounds = []
    for round in rounds:
        if model in [Ridge, Lasso]:
            reg = model(lmbda = lmbda)
        else: 
            reg = model()

        BS = call_Bootstrap(y, X, degree=degree, rounds=round, reg=reg)
        Bootstrap_list_rounds.append(BS)

        if model == LinearRegression and round in [30,102,408,606] and hist:
            plot_hist_of_bootstrap(BS, degree, model)

    return Bootstrap_list_rounds


def bootstrap_across_degrees(D, model, round, degrees, lmbdas=None):
    """Function for doing the analysis of bootstrap across degrees given a certain model and number of bootstrap rounds. Used to calculate the necesities for plotting test/train mse and bias-variance-decomposition. 

    Args:
        model (_type_): Model for linear regression
        round (int): number of bootstrap rounds
        degrees (ndarray): degress on which to preform the analysis
        lmbdas (ndarray, optional): array with same size as degrees. Each element of lmbdas corresponds to a polynomial degree. For Ridge and Lasso.  Defaults to None.
    """
    y, X = D.unpacked()

    Bootstrap_list = []
    for i, deg in enumerate(degrees): 
        if model in [Ridge, Lasso]:
            reg = model(lmbda = lmbdas[i])
        else: 
            reg = model()

        BS = call_Bootstrap(y, X, degree=deg, rounds=round, reg=reg)
        Bootstrap_list.append(BS)

    return Bootstrap_list


def bootstrap_across_lmbdas(D, lmbdas, model, round, degree):
    """Function for doing the analysis of bootstrap across lambdas given a certain model (of a certain degree) and number of bootstrap rounds. Used to calculate the necesities for plotting bias-variance-decomposition across lambdas. 

    Args:
        lmbdas (ndarray): array of lambda values. 
        model (_type_): Model for linear regression
        round (int): number of bootstrap rounds
        degrees (int): polynomial degree 
    """
    y, X = D.unpacked()

    Bootstrap_list = []
    for lmbda in lmbdas: 
        reg = model(lmbda = lmbda)
        BS = call_Bootstrap(y, X, degree=degree, rounds=round, reg=reg)
        Bootstrap_list.append(BS)

    return Bootstrap_list


def plot_comparison(BS_lists, models=[LinearRegression, Ridge, Lasso]):
    #Plotting the train/test for various models
    sns.set_style('darkgrid')
    for i, Bootstrap_list in enumerate(BS_lists):
        #Extracting the mean value for each degree:
        mse_train = [BS.mse_train for BS in Bootstrap_list]
        mse_test = [BS.mse_test for BS in Bootstrap_list]

        plt.plot(degrees, mse_train, label=f'Train MSE: {model_names[models[i]]}', c=colors[i], lw=2.5, ls='dashed')
        plt.plot(degrees, mse_test, label=f'Test MSE: {model_names[models[i]]}', c=colors[i], lw=2.5)

    plt.ylim([0,mse_test[0]*2])
    plt.xlabel('Polynomial degree', fontsize=fontsize_lab)
    plt.ylabel('MSE value', fontsize=fontsize_lab)
    plt.legend(fontsize = fontsize_leg)
    plt.title(f'Train and Test MSE: All Models', fontsize=fontsize_tit)
    plt.savefig(make_figs_path(f'BS_train_test_mse_all_models.pdf'), dpi=300)
    plt.show()


if __name__ == "__main__":
    D = make_FrankeFunction(n=600, random_state=321, noise_std=0.1)
    # D = load_Terrain()

    rounds = np.arange(30, 1000+1, (1001-30)//100)
    degrees = np.arange(1, 15+1)
    round = 400

###
#All Models
###
    BS_lists_rounds = []
    BS_lists_deg = []
    models = [LinearRegression, Ridge, Lasso]
    for model in models:
        if model in [Ridge, Lasso]:
            degrees_grid, lmbdas_grid, MSEs = load(model_names[model].lower()+'_grid')
            optimal_lmbdas = lmbdas_grid[0, np.argmin(MSEs, axis=1)]
            optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)]
            optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]
        else:
            optimal_degree = 7
            optimal_lmbda = None
            optimal_lmbdas = None

        #across rounds:
        # BS_list_rounds = bootstrap_across_rounds(D, model, rounds, degree=optimal_degree, lmbda=optimal_lmbda) #Shows that round=400 gives the stabilized state
        # BS_lists_rounds.append(BS_list_rounds)
        # plot_mse_across_rounds(BS_list_rounds, rounds, model) #FLAG HERE

        #across degrees:
        BS_list_deg = bootstrap_across_degrees(D, model, round, degrees, lmbdas=optimal_lmbdas)
        BS_lists_deg.append(BS_list_deg)
        # plot_train_test_mse(BS_list_deg, degrees, model)
        # plot_bias_var(BS_list_deg, degrees, model)

    plot_comparison(BS_lists_deg)
    exit()


###
#For various n's
###
#Wish to plot the train/test and bias-var decomposition for various n's. Should  be in the same plot?
    models = [LinearRegression, Ridge, Lasso]
    for model in models: 
        if model == Lasso:
            ns = [60, 600]
        else:
            ns = [60, 600, 1500]
        for n in ns:
            D = make_FrankeFunction(n=n, random_state=321, noise_std=0.1)
            y, X = D.unpacked()

            BS_list = bootstrap_across_degrees(model, round, degrees)
            plot_bias_var(BS_list, degrees, model, title=f"Bias-Variance Decomposition: {model_names[model]}, {n} data points", filename=f"BS_Bias_var_decomp_{model_names[model]}_{n}_data_points.pdf")


###
#For two lambdas
###
    degrees = np.arange(1, 20+1)

    models = [Ridge, Lasso]
    for model in models:
        degrees_grid, lmbdas_grid, MSEs = load(model_names[model].lower()+'_grid')
        optimal_lmbda = lmbdas_grid[np.unravel_index(np.argmin(MSEs), MSEs.shape)] * np.ones_like(degrees)
        bad_boy_lmbda = 1e0 * np.ones_like(degrees)
        BS_R_lmbda1 = bootstrap_across_degrees(D, model, round, degrees, lmbdas=optimal_lmbda)
        BS_R_lmbda2 = bootstrap_across_degrees(D, model, round, degrees, lmbdas=bad_boy_lmbda)
        plot_bias_var_2lmbda(BS_R_lmbda1, BS_R_lmbda2, optimal_lmbda, bad_boy_lmbda, degrees, model)

###
#Across lambdas
###
    models = [Ridge]
    lmbdas = np.logspace(-9,0,15)
    for model in models:
        degrees_grid, lmbdas_grid, MSEs = load(model_names[model].lower()+'_grid')
        optimal_degree = (np.unravel_index(np.argmin(MSEs), MSEs.shape))[1]
        BS_list_lam = bootstrap_across_lmbdas(D, lmbdas, model, round=400, degree=optimal_degree)
        plot_bias_var_lmbdas(BS_list_lam, lmbdas, model)






    









    

    

    


