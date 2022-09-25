import numpy as np
from cvxopt import matrix, solvers
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from sknotlearn.l1regls import l1regls

class Model:
	"""
	Base model class for linear models (OLS, Ridge, Lasso). Implements basic features such as prediction,
	in addition to some metric such as mse and r^2 score. This class is not suppose to be used for
	actual fitting.
	"""
	def __init__(self, method="SVD"):
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

	def fit(self, X, y):
		"""
		Main function to preform fitting. Calls one of the methods available in self.methods_.
	
		Args:
			X: (np.array), Design matrix to use for coefficient fitting.
			y: (np.array), Target vector which model should use the design matrix for prediction.
		Returns:
			self: (Model), Returns itself. Used to make .Model.fit(X, y) possible in same fashion as sklearn.
		"""
		return self.methods_[self.method_](X, y)

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

	def mse(self, y, y_pred):
		"""
		Calculates the mean squared error based on true y values and predicted y values. 

		Args:
			y: (np.array), True y values for mse evaluation
			y_pred: (np.array), Predicted y values for mse evaluation
		Returns:
			mse: (float), Mean squared error calculated using y and y_pred
		"""
		return np.mean((y-y_pred)**2)

	def r2_score(self, y, y_pred):
		"""
		Calculates the r-squared score based on true y values and predicted y values. 

		Args:
			y: (np.array), True y values for r-squared evaluation
			y_pred: (np.array), Predicted y values for r-squared evaluation
		Returns:
			r2_score: (float), R-squared calculated using y and y_pred
		"""
		y_bar = np.mean(y)
		return 1 - np.sum((y-y_pred)**2)/np.sum((y-y_bar)**2)

	def bias2(self, y:np.ndarray, y_pred:np.ndarray) -> float:
		"""_summary_

		Args:
			y (np.ndarray): _description_
			y_pred (np.ndarray): _description_

		Returns:
			float: _description_
		"""
		return np.mean((y - np.mean(y_pred))**2)

	def var(self, y:np.ndarray, y_pred:np.ndarray) -> float:
		"""
		Calculates the variance of y_pred. Takes in y also because it is used as bias/mse (especially in Bootstrap). 
		This could be done better.

		Args:
			y (np.ndarray): _description_
			y_pred (np.ndarray): _description_

		Return:
			float: 
		"""
		return np.var(y_pred)

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
	def fit_matrix_inv(self, X, y): 
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_svd_decomp(self, X, y):
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_matrix_psuedo_inv(self, X, y):
		raise NotImplementedError("Model base does not implement useful stuff")
	
	def fit_cost_min(self, X, y):
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

	def fit_matrix_inv(self, X, y):	
		"""
		Estimates coefficients based on default matrix inversion of X.T @ X.
		This matrix might be singular, then using the pseudo inverse method might help. 

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""
		self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
		return self

	def fit_matrix_psuedo_inv(self, X, y):
		"""
		Estimates coefficients based on inversion of X.T @ X using the pseudo inverse. This
		in the cases X.T @ X is singular.  

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""
		self.coef_ = np.linalg.pinv(X.T @ X) @ X.T @ y
		return self

	def fit_svd_decomp(self, X, y):
		"""
		Estimates coefficients based on SVD decomposition. This does NOT work for a singular matrix.

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""		
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
	def __init__(self, lmbda, **kwargs):
		"""
		Simply call Models constructor. Key-word arguments are passed to Model constructor.  
		"""
		self.lmbda = lmbda
		Model.__init__(self, **kwargs)

		self.methods_ = {
			"INV" : self.fit_matrix_inv,
			"cMIN" : self.fit_cost_min,
		}

	def fit_matrix_inv(self, X, y):	
		"""
		Estimates coefficients based on default matrix inversion of X.T @ X + I.
		Matrix is never singular, so pseudo inversion not supported.

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""
		_, p = X.shape
		self.coef_ = np.linalg.inv(X.T @ X + self.lmbda*np.eye(p)) @ X.T @ y
		return self

	def fit_svd_decomp(self, X, y):
		"""
		Estimates coefficients based on SVD decomposition.

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""		
		U, S, VT = np.linalg.svd(X, full_matrices=False)

		_, p = X.shape
	
		tol = 1e-12
		S = S[np.where(S > tol)]
		S_inv = np.zeros((p,p))	
		for i in range(len(S)):
			S_inv[i,i] = 1/S[i]

		self.coef_ = VT.T @  S_inv @ U.T @ y
		return self

	def fit_cost_min(self, X, y):	
		"""

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""
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

		self.coef_ = solvers.cp(F)['x']
		return self

class Lasso(Model):
	"""
	Implementation of Ridge Regression. Can preform coefficient estimation using both direct matrix inversion of X.T @ X 
	and trough SVD decomposition.  
	"""
	def __init__(self, lmbda, **kwargs):
		"""
		Simply call Models constructor. Key-word arguments are passed to Model constructor.  
		"""
		if lmbda == 0:
			self.fit_cost_min = LinearRegression.fit_matrix_psuedo_inv # if lmbda=0 revert to OLS.
		self.lmbda = lmbda
		Model.__init__(self, method="cMIN", **kwargs)

		self.methods_ = {
			"cMIN" : self.fit_cost_min
		}

	def fit_cost_min(self, X, y):	
		"""

		Args:
			X: np.array, Design matrix used for coefficient fitting
			y: np.array, Target values used for fitting.
		"""
		A = matrix(X/np.sqrt(self.lmbda))
		b = matrix(y/np.sqrt(self.lmbda))
		self.coef_ = np.array(l1regls(A, b))[:,0]
		return self

solvers.options['show_progress'] = False # suppresses cvxopt messages