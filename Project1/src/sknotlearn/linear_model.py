import numpy as np

class Model:
	"""
	Base model class for linear models (OLS, Ridge, Lasso). Implements basic features such as prediction,
	in addition to some metric such as mse and r^2 score. This class is not suppose to be used for
	actual fitting.
	"""
	def __init__(self, method="SVD"):
		self.method_ = method

		self.methods_ = {
			"INV": self.fit_matrix_inv,
			"SVD": self.fit_svd_decomp
		}

		self.coef_ = None

	def fit(self, X, y):
		return self.methods_[self.method_](X, y)

	def predict(self, X):
		return X @ self.coef_

	def mse(self, y, y_pred):
		return np.mean((y-y_pred)**2)

	def r2_score(self, y, y_pred):
		y_bar = np.mean(y)
		return 1 - np.sum((y-y_pred)**2)/np.sum((y-y_bar)**2)

	def fit_matrix_inv(self, X, y):
		raise NotImplementedError("Model base does not implemen usefull stuff")

	def fit_svd_decomp(self, X, y):
		raise NotImplementedError("Model base does not implemen usefull stuff")

class LinearRegression(Model):
	"""
	Implementation of OLS. Can preform coefficient estimation using both direct matrix inversion of X.T @ X 
	and trough SVD decomposition.  
	"""
	def __init__(self, **kwargs):
		Model.__init__(self, **kwargs)

	def fit_matrix_inv(self, X, y):	
		self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
		return self

	def fit_svd_decomp(self, X, y):
		U, S, VT = np.linalg.svd(X, full_matrices=False)

		self.coef_ = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
		
		return self
