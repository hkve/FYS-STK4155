import numpy as np
from time import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.datasets import make_FrankeFunction

# # Returns mean of bootstrap samples 
# # Bootstrap algorithm
# def bootstrap(data, datapoints):
#     t = np.zeros(datapoints)
#     n = len(data)
#     # non-parametric bootstrap         
#     for i in range(datapoints):
#         t[i] = np.mean(data[np.random.randint(0,n,n)])
#     # analysis    
#     print("Bootstrap Statistics :")
#     print("original           bias      std. error")
#     print("%8g %8g %14g %15g" % (np.mean(data), np.std(data),np.mean(t),np.std(t)))
#     return t

# # We set the mean value to 100 and the standard deviation to 15
# mu, sigma = 100, 15
# datapoints = 10000
# # We generate random numbers according to the normal distribution
# x = mu + sigma*np.random.randn(datapoints)
# # bootstrap returns the data sample                                    
# t = bootstrap(x, datapoints)


# # the histogram of the bootstrapped data (normalized data if density = True)
# n, binsboot, patches = plt.hist(t, 50, density=True, facecolor='red', alpha=0.75)
# # add a 'best fit' line  
# y = norm.pdf(binsboot, np.mean(t), np.std(t))
# lt = plt.plot(binsboot, y, 'b', linewidth=1)
# plt.xlabel('x')
# plt.ylabel('Probability')
# plt.grid(True)
# plt.show()

def bootstrap(y_train, x_train, y_test, x_test, rounds, method_='SVD'):
    """Will take in x_train which is chosen multiple times 

    Args:
        y_train (_type_): _description_
        x_train (_type_): _description_
        y_test (_type_): _description_
        x_test (_type_): _description_
        rounds (int): rounds of bootstrapping the dataset 
        method_ (str, optional): _description_. Defaults to 'SVD'.
    """
    
    mse_train_values = np.zeros(rounds)
    mse_test_values = np.zeros(rounds)
    n = len(x_train)
    for i in range(rounds): 
        indices = np.random.randint(0,n,n)
        x_train_boot = x_train[indices]
        y_train_boot = y_train[indices]

        reg = LinearRegression(method=method_).fit(x_train_boot, y_train_boot)
        y_train_pred = reg.predict(x_train_boot)
        y_test_pred = reg.predict(x_test)

        mse_train_values[i] = reg.mse(y_train_pred, y_train_boot)
        mse_test_values[i] = reg.mse(y_test_pred, y_test)
    
    return mse_train_values, mse_test_values


if __name__ == "__main__":
    X, y = make_FrankeFunction(n=600, uniform=True, random_state=321)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=6)
    X_poly = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=321)

    mse_train, mse_test = bootstrap(y_train, X_train, y_test, X_test, rounds=1000, method_='SVD')

    bins = 10
    plt.hist(mse_train, bins)
    plt.hist(mse_test, bins=100, alpha=0.6)
    plt.show()

