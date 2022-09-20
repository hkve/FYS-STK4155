# This is horrible, but if you find a better way please tell (and explain) to me how
import context

from sknotlearn.linear_model import LinearRegression
import numpy as np

def test_LinearRegression():
	x = np.random.uniform(low=-5, high=5, size=40)
	beta = np.array([1, 0.25, -1.5, 3])

	y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)
	X = np.c_[np.ones_like(x), x, x**2, x**3]
	lin_INV = LinearRegression(method="INV").fit(X, y)
	lin_SVD = LinearRegression(method="SVD").fit(X, y)

	tol = 1e-12
	assert np.sum(np.abs(beta-lin_INV.coef_)) < tol, "Error in full LinearRegression test using matrix inversion" 
	assert np.sum(np.abs(beta-lin_SVD.coef_)) < tol, "Error in full LinearRegression test using SVD decomposition" 

def test_mse():
	lin = LinearRegression()
	y_pred = np.array([1,2,3])
	y = np.array([3,2,1])

	expected = 8/3
	calculated = lin.mse(y, y_pred)

	tol = 1e-12
	assert np.abs(expected-calculated) < tol, "Error in LinearRegression mse calculations"


def test_r2():
	lin = LinearRegression()
	y_pred = np.array([1,2,3])
	y = np.array([2,1,3])

	calculated = lin.r2_score(y, y_pred)
	expected = 0

	tol = 1e-12
	assert np.abs(expected-calculated) < tol, "Error in LinearRegression mse calculations"