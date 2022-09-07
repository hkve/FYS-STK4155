import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.datasets import make_FrankeFunction



def solve_a(n=1000, train_size=0.8, random_state=123):
	"""
	Function to preform what I assume task b is asking. R2 and MSE (test and train) for polynomials (1,5)

	1. Generate x, y data from Franke Function 
	2. Scale design matrix per column  
	3. Make polynomial features using the x and y values from design matrix
	4. Calculate measures and save coefs
	5. plot
	"""
	X, y = make_FrankeFunction(n=n, uniform=True, random_state=random_state)

	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	
	degrees = np.arange(1, 5+1)
	mse_train = np.zeros_like(degrees, dtype=float)
	mse_test = np.zeros_like(degrees, dtype=float)
	r2_train = np.zeros_like(degrees, dtype=float)
	r2_test = np.zeros_like(degrees, dtype=float)
	betas = {} 

	for i, degree in enumerate(degrees):
		poly = PolynomialFeatures(degree=degree)

		X_poly = poly.fit_transform(X)
		

		X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=random_state)

		reg = LinearRegression(method="SVD").fit(X_train, y_train)

		y_train_pred = reg.predict(X_train)
		y_test_pred = reg.predict(X_test)

		mse_train[i] = reg.mse(y_train_pred, y_train)
		mse_test[i] = reg.mse(y_test_pred, y_test)

		r2_train[i] = reg.r2_score(y_train_pred, y_train)
		r2_test[i] = reg.r2_score(y_test_pred, y_test)

		betas[degree] = reg.coef_

	fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,6))
	axes[0].plot(degrees, mse_train, c="b", label="train")
	axes[0].plot(degrees, mse_test, c="r", label="test")
	axes[0].set(xlabel="degree", ylabel="MSE")
	axes[0].legend()


	axes[1].plot(degrees, r2_train, c="b", label="train")
	axes[1].plot(degrees, r2_test, c="r", label="test")
	axes[1].set(xlabel="degree", ylabel=r"$R^2$")
	axes[1].legend()

	colors = plt.cm.get_cmap("hsv", 21)
	ax = axes[2]
	ax.set(xlabel="degree", ylabel=r"$\hat{\beta}$")
	
	for degree, beta in betas.items():
		for i, coef in enumerate(beta):
			ax.scatter(degree, coef, color=colors(i))

	fig.tight_layout()
	plt.show()

solve_a()
