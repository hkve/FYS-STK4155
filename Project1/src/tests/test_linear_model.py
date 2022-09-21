# This is horrible, but if you find a better way please tell (and explain) to me how
import context

import sknotlearn.linear_model
import numpy as np
np.random.seed(1232)

import sklearn.linear_model
from sklearn.preprocessing import StandardScaler

def test_LinearRegression():
	x = np.random.uniform(low=-5, high=5, size=40)
	beta = np.array([1, 0.25, -1.5, 3])

	y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)
	X = np.c_[np.ones_like(x), x, x**2, x**3]	
	lin_INV = sknotlearn.linear_model.LinearRegression(method="INV").fit(X, y)
	lin_SVD = sknotlearn.linear_model.LinearRegression(method="SVD").fit(X, y)

	tol = 1e-12
	assert np.sum(np.abs(beta-lin_INV.coef_)) < tol, "Error in full LinearRegression test using matrix inversion" 
	assert np.sum(np.abs(beta-lin_SVD.coef_)) < tol, "Error in full LinearRegression test using SVD decomposition" 

def test_Ridge():
	x = np.random.uniform(low=-5, high=5, size=40)
	beta = np.array([1, 0.25, -1.5, 3])

	y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)
	X = np.c_[np.ones_like(x), x, x**2, x**3]

	X = StandardScaler().fit_transform(X)

	lmbdas = [0.1, 0.5, 1, 2]

	for lmbda in lmbdas:
		r1 = sknotlearn.linear_model.Ridge(lmbda=lmbda, method="INV").fit(X,y)
		r2 = sklearn.linear_model.Ridge(alpha=lmbda).fit(X,y)

		tol = 1e-12
		assert np.sum(np.abs(r1.coef_ - r2.coef_)) < tol, f"Error in Ridge regression using matrix inversion for {lmbda = }"


def test_mse():
	lin = sknotlearn.linear_model.LinearRegression()
	y_pred = np.array([1,2,3])
	y = np.array([3,2,1])

	expected = 8/3
	calculated = lin.mse(y, y_pred)

	tol = 1e-12
	assert np.abs(expected-calculated) < tol, "Error in LinearRegression mse calculations"


def test_r2():
	lin = sknotlearn.linear_model.LinearRegression()
	y_pred = np.array([1,2,3])
	y = np.array([2,1,3])

	calculated = lin.r2_score(y, y_pred)
	expected = 0

	tol = 1e-12
	assert np.abs(expected-calculated) < tol, "Error in LinearRegression mse calculations"