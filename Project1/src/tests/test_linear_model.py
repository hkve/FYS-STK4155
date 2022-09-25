# This is horrible, but if you find a better way please tell (and explain) to me how
import context

import sknotlearn.linear_model
from sknotlearn.data import Data
import numpy as np
np.random.seed(321)

import sklearn.linear_model
from sklearn.preprocessing import StandardScaler

def test_LinearRegression():
	x = np.random.uniform(low=-5, high=5, size=40)
	beta = np.array([1, 0.25, -1.5, 3])

	y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)
	X = np.c_[np.ones_like(x), x, x**2, x**3]	
	data = Data(y, X)
	lin_INV = sknotlearn.linear_model.LinearRegression(method="INV").fit(data)
	lin_SVD = sknotlearn.linear_model.LinearRegression(method="SVD").fit(data)

	tol = 1e-12
	assert np.sum(np.abs(beta-lin_INV.coef_)) < tol, "Error in full LinearRegression test using matrix inversion" 
	assert np.sum(np.abs(beta-lin_SVD.coef_)) < tol, "Error in full LinearRegression test using SVD decomposition" 

def test_Ridge():
	x = np.random.uniform(low=-5, high=5, size=40)
	beta = np.array([1, 0.25, -1.5, 3])

	y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)
	X = np.c_[np.ones_like(x), x, x**2, x**3]
	data = Data(y, X)

	# X = StandardScaler().fit_transform(X)
	scaled_data = data.scaled(scheme="Standard")

	lmbdas = [0.1, 0.5, 1, 2]

	for lmbda in lmbdas:
		r1 = sknotlearn.linear_model.Ridge(lmbda=lmbda, method="INV").fit(scaled_data)
		y, X = scaled_data.unpacked()
		r2 = sklearn.linear_model.Ridge(alpha=lmbda).fit(X, y)

		tol = 1e-12
		assert np.sum(np.abs(r1.coef_ - r2.coef_)) < tol, f"Error in Ridge regression using matrix inversion for {lmbda = }"