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
from sknotlearn.linear_model import LinearRegression
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.resampling import Bootstrap
from sknotlearn.data import Data

""" 
With the bias and var calculated across the bootstrap
(With data-class implemented)
"""

def plot_hist_of_bootstrap(Bootstrap_, degree):
    """Generate the histogram of the mse-values of a certain model with a given number 
    of bootstrap rounds

    Args:
        Bootstraps (_type_): bootstrap of a certain degree
    """
    mse_train = Bootstrap_.mse_train_values
    mse_test = Bootstrap_.mse_test_values

    find_bins = lambda arr, times=250: np.abs(int((np.max(arr)-np.min(arr))*times))
    plt.hist(mse_train, label='mse for training data')
    plt.hist(mse_test, alpha=0.75,  label='mse for test data') #Cannot use density=True because MSE
    plt.xlabel('MSE')
    plt.ylabel('Probability')
    plt.title(f'Model of degree {degree}: Results of MSE when bootstrapping {Bootstrap_.rounds} times')
    plt.legend()
    plt.show()

def plot_bias_var(Bootstrap_list, degrees):
    mse = [BS.mse_test for BS in Bootstrap_list]
    bias = [BS.bias_test for BS in Bootstrap_list]
    var = [BS.var_test for BS in Bootstrap_list]
    proj_mse = [BS.projected_mse for BS in Bootstrap_list]

    sns.set_style('darkgrid')
    plt.plot(degrees, mse, label='MSE', c=sns.color_palette('husl')[-1], lw=2)
    plt.plot(degrees, bias, label=r'Bias$^2$', c=sns.color_palette('husl')[-3], lw=2)
    plt.plot(degrees, var, label='Variance', c=sns.color_palette('husl')[-2], lw=2)
    plt.plot(degrees, proj_mse, '--', label='Projected mse', c=sns.color_palette('colorblind')[1], lw=2)
    plt.legend()
    plt.show()


def plot_mse(Bootstrap_list, degrees):
    mse_train = [BS.mse_train for BS in Bootstrap_list]
    mse_test = [BS.mse_test for BS in Bootstrap_list]

    plt.plot(degrees, mse_train)
    plt.plot(degrees, mse_test)
    plt.show()


if __name__ == "__main__":
    D = make_FrankeFunction(n=600, uniform=True, random_state=321, noise_std=0.1)
    D.scaled()
    y, X = D.unpacked()

    print(len(sns.color_palette('husl')))



    # #For various rounds:
    # #Need to fix bins as well as subplots
    # rounds = np.arange(100, 1000+1, 200)
    # Bootstrap_list_rounds = []
    # for round in rounds:
    #     poly = PolynomialFeatures(degree=7)
    #     X_poly = poly.fit_transform(X)

    #     data = Data(y, X_poly)

    #     data_train, data_test = data.train_test_split(ratio=3/4, random_state=321)
    #     reg = LinearRegression()

    #     BS = Bootstrap(reg, data_train, data_test, random_state = 321, rounds=round)
    #     Bootstrap_list_rounds.append(BS)
    #     plot_hist_of_bootstrap(BS, 7)

   
    #For various degrees
    degrees = np.arange(1, 12+1)
    Bootstrap_list = []
    for deg in degrees: 
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)

        data = Data(y, X_poly)

        data_train, data_test = data.train_test_split(ratio=3/4, random_state=321)
        reg = LinearRegression()

        BS = Bootstrap(reg, data_train, data_test, random_state = 321, rounds=70)
        Bootstrap_list.append(BS)

    # plot_hist_of_bootstrap(Bootstrap_list[7], 7)
    # plot_mse(Bootstrap_list, degrees)
    plot_bias_var(Bootstrap_list, degrees)

    

    


"""
OLD!!!:
"""

exit()
""" 
Below here is everything from when we thought we could calculate 
everything in the class.........
"""


def hastie_2_11_ex_c(Bootstrap_list, degrees):
    MSE_train = [BS.scores_["train_mse"] for BS in Bootstrap_list]
    MSE_test = [BS.scores_["test_mse"] for BS in Bootstrap_list]
    plt.plot(degrees, np.mean(MSE_train, axis=1), label='train')
    plt.plot(degrees, np.mean(MSE_test, axis=1), label='test')
    plt.show()

def plot_bias_var(Bootstrap_list, degrees):
    MSE = [BS.scores_["test_mse"] for BS in Bootstrap_list]
    bias =  [BS.scores_["test_bias2"] for BS in Bootstrap_list]
    var =  [BS.scores_["test_var"] for BS in Bootstrap_list]

    plt.plot(degrees, np.mean(MSE, axis=1), label="Error")
    plt.plot(degrees, np.mean(bias, axis=1), label="Bias")
    plt.plot(degrees, np.mean(var, axis=1), label="Variance")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    X, y = make_FrankeFunction(n=600, uniform=True, random_state=4110)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    degrees = np.arange(1, 12+1)
    Bootstrap_list = []
    for deg in degrees: 
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        data = Data(y, X_poly)
        data_train, data_test = data.train_test_split(ratio=3/4, random_state=4110)
        reg = LinearRegression()

        BS = Bootstrap(reg, data_train, data_test, random_state = 4110, rounds=20, scoring=("mse", "bias2", "var"))
        Bootstrap_list.append(BS)

    plot_hist_of_bootstrap(Bootstrap_list[4].scores_["train_mse"], Bootstrap_list[4].scores_["test_mse"], 4)

    hastie_2_11_ex_c(Bootstrap_list, degrees)

    plot_bias_var(Bootstrap_list, degrees)
















exit()
""" 
Below is from before the class was implemented 
"""
# def mse_from_bootstrap(Bootstrap_list):
#     """Generate the mse values from the bootstrapped 
#       Might not work

#     Args:
#         Bootstraplist (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     n = len(Bootstrap_list)

#     mse_test_values = np.zeros(n)
#     mse_train_values = np.zeros(n)

#     for i in range(n):
#         BS = Bootstrap_list[i]

#         mse_test_values[i] = np.mean((BS.y_test_values - BS.y_test_pred_values)**2)
#         mse_train_values[i] = np.mean((BS.y_train_boot_values - BS.y_train_pred_values)**2)

#     return mse_test_values, mse_train_values


def bootstrap(y_train, x_train, y_test, x_test, rounds, method='SVD'):
    """Will take in x_train which is chosen multiple times 

    NB: It is important to keep the testdata constant. 

    Args:
        y_train (_type_): _description_
        x_train (_type_): _description_
        y_test (_type_): _description_
        x_test (_type_): _description_
        rounds (int): rounds of bootstrapping the dataset 
        method_ (str, optional): _description_. Defaults to 'SVD'.
    """
    
    n = len(x_train)
    y_test_pred_values = np.empty((len(y_test),rounds))
    y_test_values = np.empty((len(y_test),rounds))
    mse_train_values = np.zeros(rounds)
    mse_test_values = np.zeros(rounds)
    
    for i in range(rounds): 
        indices = np.random.randint(0,n,n)
        x_train_boot = x_train[indices]
        y_train_boot = y_train[indices]

        reg = LinearRegression(method=method).fit(x_train_boot, y_train_boot)
        y_train_pred = reg.predict(x_train_boot)
        y_test_pred = reg.predict(x_test)

        mse_train_values[i] = reg.mse(y_train_pred, y_train_boot)
        mse_test_values[i] = reg.mse(y_test_pred, y_test)

        y_test_pred_values[:,i] = y_test_pred
        y_test_values[:,i] = y_test
    
    return mse_train_values, mse_test_values, y_test_pred_values, y_test_values


def mse_from_bootstrap(n=600, uniform=True, random_state=321, degrees=np.arange(1, 12+1), rounds=600, method='SVD'):
    X, y = make_FrankeFunction(n=n, uniform=uniform, random_state=random_state)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    mse_train_list = []
    mse_test_list = []
    # for deg in degrees: 
    #     poly = PolynomialFeatures(degree=deg)
    #     X_poly = poly.fit_transform(X)
    #     X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=random_state)

    #     mse_train, mse_test, _, _ = bootstrap(y_train, X_train, y_test, X_test, rounds=rounds, method=method)
    #     mse_train_list.append(mse_train)
    #     mse_test_list.append(mse_test)

    #Nanna:
    Bootstrap_list = []
    for deg in degrees: 
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=random_state)
        reg = LinearRegression()

        BS = Bootstrap(X_train, X_test, y_train, y_test, reg, 123)
        BS(100)
        Bootstrap_list.append(BS)

    
    return mse_train_list, mse_test_list

#Nanna:
def plot(bootsraps):
    MSE = [BS.trainMSE for BS in bootsraps]


def plot_hist_of_bootstrap(mse_train, mse_test, degree, rounds=600):
    find_bins = lambda arr, times=25000: int((np.max(arr)-np.min(arr))*times)
    plt.hist(mse_train, bins=find_bins(mse_train), label='mse for training data')
    plt.hist(mse_test, alpha=0.75, bins=find_bins(mse_test),  label='mse for test data') #Cannot use density=True because MSE
    plt.xlabel('MSE')
    plt.ylabel('Probability')
    plt.title(f'Model of degree {degree}: Results of MSE when bootstrapping {rounds} times')
    plt.legend()
    plt.show()


def hastie_2_11_ex_c(mse_train_list, mse_test_list, degrees):
    #Could have saved degrees as a global variable 
    plt.plot(degrees, np.mean(mse_train_list, axis=1), label='train')
    plt.plot(degrees, np.mean(mse_test_list, axis=1), label='test')
    plt.show()


def bias_var(n=600, uniform=True, random_state=123, degrees=np.arange(1, 12+1)):
    """NB: This is not correct!!!
    """
    X, y = make_FrankeFunction(n=n, uniform=uniform, random_state=random_state)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    poly_degrees = np.zeros(degrees.shape[0])
    error = np.zeros(degrees.shape[0])
    error2 = np.zeros(degrees.shape[0])
    bias = np.zeros(degrees.shape[0])
    variance = np.zeros(degrees.shape[0])

    for i, deg in enumerate(degrees): 
        #Make the model + data
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=random_state)

        #Fit the model through bootstrap
        mse_train, mse_test, y_test_pred_val, y_test_val = bootstrap(y_train, X_train, y_test, X_test, 60)

        #Calculate the relevant values of the bias_var_trade_off:
        poly_degrees[i] = deg
        error[i] = np.mean(mse_train)
        error2[i] = np.mean(mse_test)
        bias[i] = np.mean((y_test_val - np.mean(y_test_pred_val, axis=1, keepdims=True))**2)
        variance[i] = np.mean(np.var(y_test_pred_val))
    
    plt.plot(poly_degrees, error, label='Error')
    plt.plot(poly_degrees, error2,'--', label='Error2')
    plt.plot(poly_degrees, bias, label='Bias')
    plt.plot(poly_degrees, variance, label='Variance')
    plt.legend()
    plt.show()
        


if __name__ == "__main__":
    # mse_train_list, mse_test_list = mse_from_bootstrap()
    # hastie_2_11_ex_c(mse_train_list, mse_test_list, degrees=np.arange(1, 12+1))

    # i = 0
    # plot_hist_of_bootstrap(mse_train_list[i], mse_test_list[i], degree=i+1) 


    # ### Bias_var:
    bias_var()



    


# Play around with the number of bootstrap rounds




