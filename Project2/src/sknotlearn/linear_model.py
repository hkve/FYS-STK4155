import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False # suppresses cvxopt messages

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l1regls import l1regls
from data import Data

class Model:
	"""
	Base model class for linear models (OLS, Ridge, Lasso). Implements basic features such as prediction,
	in addition to some metric such as mse and r^2 score. This class is not suppose to be used for
	actual fitting.
	"""
	def __init__(self, method="pINV"):
		"""
		Default constructor takes optional method argument, deciding how the model coefficients should
		be fitted. 

		Args:
			method: (str), optional. Name of method used to perform coefficient fitting
		"""
		self.methods_ = {
			"INV": self.fit_matrix_inv,
			"pINV": self.fit_matrix_psuedo_inv,
			"SVD": self.fit_svd_decomp,
			"cMIN": self.fit_cost_min,
		}

		self.metrics_ = {
			"mse": self.mse,
			"r2_score": self.r2_score,
			"bias2": self.bias2,
			"var": self.var
		}
		
		assert method in self.methods_.keys(), f"{method = } is not a valid method. Please choose between {self.methods_.keys()}"
		self.method_ = method

		self.coef_ = None

	def fit(self, data):
		"""
		Main function to preform fitting. Calls one of the methods available in self.methods_.
	
		Args:
			data: (Data), Data-object containing design matrix and target vectors for fitting the model.
		Returns:
			self: (Model), Returns itself. Used to make .Model.fit(X, y) possible in same fashion as sklearn.
		"""
		return self.methods_[self.method_](data)

	def predict(self, X):
		"""
		Function to preform prediction based on a design matrix X

		Args:
			X: (np.array), Design matrix
		Returns:
			y: (np.array), Prediction based on the passed design matrix
		"""
		
		assert self.coef_ is not None, "The model must be fitted before preforming a prediction"

		return X @ self.coef_

	def mse(self, data:Data):
		"""
		Calculates the mean squared error based on true y values and predicted y values. 

		Args:
			data (Data): Data object containing target y-values and corresponding features
		Returns:
			mse (float): Mean squared error calculated using y and y_pred
		"""
		y_target, X = data.unpacked()
		return np.mean((y_target-self.predict(X))**2)

	def r2_score(self, data:Data):
		"""
		Calculates the r-squared score based on true y values and predicted y values. 

		Args:
			data (Data): Data object containing target y-values and corresponding features
		Returns:
			r2_score: (float), R-squared calculated using y and y_pred
		"""
		y_target, X = data.unpacked()
		y_bar = np.mean(y_target)
		return 1 - np.sum((y_target-self.predict(X))**2)/np.sum((y_target-y_bar)**2)

	def bias2(self, data:Data) -> float:
		"""_summary_

		Args:
			data (Data): Data object containing target y-values and corresponding features

		Returns:
			float: _description_
		"""
		y_target, X = data.unpacked()
		return np.mean(y_target - np.mean(self.predict(X)))**2

	def var(self, data:Data) -> float:
		"""
		Calculates the variance of y_pred. Takes in y also because it is used as bias/mse (especially in Bootstrap). 
		This could be done better.

		Args:
			data (Data): Data object containing target y-values and corresponding features

		Return:
			float: 
		"""
		_, X = data.unpacked()
		return np.var(self.predict(X))

	def coef_var(self, X, noise_std):
		"""
		Calculates the variance of coef.
	
		Args:
			X: (np.array), Designe matrix used for se calculations
			y: (np.array), True values of y, y_hat is calculated based on X

		Returns
			var_coef: (np.array), Coefficient variance matrix 
		"""	
		var_coef = noise_std**2 * np.linalg.pinv(X.T @ X)
		return  var_coef

	# These functions should be implemented in classes inheriting from Model
	def fit_matrix_inv(self, data:Data): 
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_svd_decomp(self, data:Data):
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_matrix_psuedo_inv(self, data:Data):
		raise NotImplementedError("Model base does not implement useful stuff")
	
	def fit_cost_min(self, data:Data):
		raise NotImplementedError("Model base does not implement useful stuff")


class LinearRegression(Model):
	"""
	Implementation of OLS. Can preform coefficient estimation using both direct matrix inversion of X.T @ X 
	and trough SVD decomposition.  
	"""
	def __init__(self, **kwargs):
		"""
		Simply call Models constructor. Key-word arguments are passed to Model constructor.  
		"""
		Model.__init__(self, **kwargs)

	def fit_matrix_inv(self, data:Data):	
		"""
		Estimates coefficients based on default matrix inversion of X.T @ X.
		This matrix might be singular, then using the pseudo inverse method might help. 

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""
		y, X = data.unpacked()
		self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
		return self

	def fit_matrix_psuedo_inv(self, data:Data):
		"""
		Estimates coefficients based on inversion of X.T @ X using the pseudo inverse. This
		in the cases X.T @ X is singular.  

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""
		y, X = data.unpacked()
		self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
		return self

	def fit_svd_decomp(self, data:Data):
		"""
		Estimates coefficients based on SVD decomposition. This does NOT work for a singular matrix.

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""		
		y, X = data.unpacked()
		U, S, VT = np.linalg.svd(X, full_matrices=False)

		_, p = X.shape
	
		tol = 1e-12
		S = S[np.where(S > tol)]
		S_inv = np.zeros((p,p))	
		for i in range(len(S)):
			S_inv[i,i] = 1/S[i]

		self.coef_ = VT.T @  S_inv @ U.T @ y
		
		return self

class Ridge(Model):
	"""
	Implementation of Ridge Regression. Can preform coefficient estimation using both direct matrix inversion of X.T @ X 
	and trough SVD decomposition.  
	"""
	def __init__(self, lmbda, method="INV"):
		"""
		Simply call Models constructor. Key-word arguments are passed to Model constructor.  
		"""
		self.lmbda = lmbda
		Model.__init__(self, method)

		self.methods_ = {
			"INV" : self.fit_matrix_inv,
			"cMIN" : self.fit_cost_min,
		}

	def fit_matrix_inv(self, data:Data):	
		"""
		Estimates coefficients based on default matrix inversion of X.T @ X + I.
		Matrix is never singular, so pseudo inversion not supported.

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""
		y, X = data.unpacked()
		_, p = X.shape
		self.coef_ = np.linalg.inv(X.T @ X + self.lmbda*np.eye(p)) @ X.T @ y
		return self

	def fit_svd_decomp(self, data:Data):
		"""
		Estimates coefficients based on SVD decomposition.

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""		
		y, X = data.unpacked()
		U, S, VT = np.linalg.svd(X, full_matrices=False)

		_, p = X.shape
	
		tol = 1e-12
		S = S[np.where(S > tol)]
		S_inv = np.zeros((p,p))	
		for i in range(len(S)):
			S_inv[i,i] = 1/S[i]

		self.coef_ = VT.T @  S_inv @ U.T @ y
		return self

	def fit_cost_min(self, data:Data):	
		"""

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""
		y, X = data.unpacked()
		def F(beta=None, z=None):
			'''
			Follows cvxopt prescription for defining convex function to minimize.
			Args:
				beta (None, cvxopt.matrix) : values of the parameters to optimize
				z (cvxopt.matrix) : wheights for when conditions are applied to cost function.
			'''
			if beta is None:
				# this initializes the problem. Return number of conditions (0), and initial parameter guess
				return 0, matrix(np.ones_like(X[0]))
			beta = np.array(beta)[:,0] # translating cvxopt matrix into numpy array
			# cost function
			f = np.sum( (y-X@beta)**2 ) + self.lmbda * np.sum( beta**2 )
			# cost function gradient
			Df = matrix(np.array([-2*X.T@(y-X@beta) + 2*self.lmbda*beta]))
			if z is None:
				return f, Df
			# cost function Hessian, approximating Dirac delta
			H = matrix(2*z[0] * (X.T@X + self.lmbda*np.eye(X.shape[-1], X.shape[-1])))
			return f, Df, H

		self.coef_ = np.array(solvers.cp(F)['x']).ravel()
		return self

class Lasso(Model):
	"""
	Implementation of Lasso Regression. Can preform coefficient estimation using cost function minimization  
	This class uses cvxopt for minimization.
	"""
	def __init__(self, lmbda, method="cMIN"):
		"""
		Simply call Models constructor. Key-word arguments are passed to Model constructor.  
		"""
		if lmbda == 0:
			self.fit_cost_min = LinearRegression.fit_matrix_psuedo_inv # if lmbda=0 revert to OLS.
		self.lmbda = lmbda
		Model.__init__(self, method)

		self.methods_ = {
			"cMIN" : self.fit_cost_min
		}

	def fit_cost_min(self, data:Data):	
		"""

		Args:
			data (Data): Data object containing target y-values and corresponding features
		"""
		y, X = data.unpacked()
		A = matrix(X/np.sqrt(self.lmbda))
		b = matrix(y/np.sqrt(self.lmbda))
		self.coef_ = np.array(l1regls(A, b))[:,0]
		return self


def OLS_gradient(coef:np.ndarray, data:Data, idcs:np.ndarray=None) -> np.ndarray:
	if idcs is None:
		idcs = np.arange(len(data.y))
	y, X = data[idcs].unpacked()
	n = len(y)
	return (2./n) * X.T @ (X @ coef - y)

def ridge_gradient(coef:np.ndarray, data:Data, lmbda, idcs:np.ndarray=None) -> np.ndarray:
	if idcs is None:
		idcs = np.arange(len(data.y))
	y, X = data[idcs].unpacked()
	n = len(y)
	return (2./n) * X.T @ (X @ coef - y) + 2. * lmbda * coef

def lasso_gradient(coef:np.ndarray, data:Data, lmbda, idcs:np.ndarray=None) -> np.ndarray:
	if idcs is None:
		idcs = np.arange(len(data.y))
	y, X = data[idcs].unpacked()
	n = len(y)
	return (2./n) * X.T @ (X @ coef - y) + 2. * lmbda * np.sign(coef)