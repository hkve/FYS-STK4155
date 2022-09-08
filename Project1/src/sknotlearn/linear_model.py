import numpy as np

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
			method: (str), optional. Name of method used to preform coefficient fitting
		"""
		self.methods_ = {
			"INV": self.fit_matrix_inv,
			"pINV": self.fit_matrix_psuedo_inv,
			"SVD": self.fit_svd_decomp
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


	# These functions should be implemented in classes inheriting from Model
	def fit_matrix_inv(self, X, y): 
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_svd_decomp(self, X, y):
		raise NotImplementedError("Model base does not implement useful stuff")

	def fit_matrix_psuedo_inv(self, X, y):
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

		self.coef_ = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ y
		
		return self