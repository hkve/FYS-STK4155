import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.datasets import make_FrankeFunction

def plot_mse_and_r2(degrees, mse_train, mse_test, r2_train, r2_test):
	"""
	Function to plot train/test mse and r2. Simple line plot with mse on the left and r2 on the right
	"""
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
	axes[0].plot(degrees, mse_train, c="b", label="train")
	axes[0].plot(degrees, mse_test, c="r", label="test")
	axes[0].set(xlabel="degree", ylabel="MSE")
	axes[0].legend()


	axes[1].plot(degrees, r2_train, c="b", label="train")
	axes[1].plot(degrees, r2_test, c="r", label="test")
	axes[1].set(xlabel="degree", ylabel=r"$R^2$")
	axes[1].legend()

	fig.tight_layout()
	plt.show()


def plot_beta_progression(betas, powers, degrees=[1,3,5]):
	"""
	Function to plot the evolution of of fitted paramters based on different polynomial degrees. Betas and powers are 
	dicts with keys as poly power (1,2,3 ...) as created by the script. Beta values are the beta values (duh) and powers an array of (x,y) powers for
	the different parameters. 
	"""
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(nrows=1, ncols=len(degrees))

	axes[0].set(ylabel="Coef size")
	axes[1].set(xlabel=r"Coef of $x^i y^j$")

	for ax, p in zip(axes, degrees):
		# Make labels like 1, x, y, xy ...
		labels = ["1"]
		for exponents in powers[p][1:]:
			label = r"$"
			for exponent, symbol in zip(exponents, ["x", "y"]):
				if exponent > 0:
					label += symbol
				if exponent > 1:
					label += ("^{" + str(exponent) + "}")
			
			label += "$"
			labels.append(label)
		
		# Plot for degree p
		ticks = np.arange(len(labels))

		ax.set_xticks(ticks, labels)
		ax.scatter(ticks, betas[p])
		ax.set(ylim=(-0.6, 0.6))


	fig.tight_layout()
	plt.show()


def solve_a(n=1000, train_size=0.8, random_state=123):
	"""
	Function to preform what I assume task b is asking. R2 and MSE (test and train) for polynomials (1,5)

	1. Generate x, y data from Franke Function 
	2. Scale design matrix per column  
	3. Make polynomial features using the x and y values from design matrix
	4. Calculate measures and save coefs
	5. Plot mse & r2
	6. Plot chaos beta plot
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
	powers = {} 

	for i, degree in enumerate(degrees):
		poly = PolynomialFeatures(degree=degree)

		X_poly = poly.fit_transform(X)
		

		X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=random_state)

		reg = LinearRegression(method="pINV").fit(X_train, y_train)

		y_train_pred = reg.predict(X_train)
		y_test_pred = reg.predict(X_test)

		mse_train[i] = reg.mse(y_train_pred, y_train)
		mse_test[i] = reg.mse(y_test_pred, y_test)

		r2_train[i] = reg.r2_score(y_train_pred, y_train)
		r2_test[i] = reg.r2_score(y_test_pred, y_test)

		betas[degree] = reg.coef_
		powers[degree] = poly.powers_

	# Show mse and r2 score 
	#plot_mse_and_r2(degrees, mse_train, mse_test, r2_train, r2_test)
	
	plot_beta_progression(betas, powers)

if __name__ == "__main__":
	solve_a()
