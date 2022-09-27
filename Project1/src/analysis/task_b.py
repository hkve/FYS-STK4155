import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data

import utils

def plot_mse_and_r2(degrees, mse_train, mse_test, r2_train, r2_test, filename=None):
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
	if filename is not None: plt.savefig(utils.make_figs_path(filename), dpi=300)

	plt.show()


def plot_beta_progression(betas, betas_se, powers, degrees=[1,3,5], filename=None):
	"""
	Function to plot the evolution of of fitted paramters based on different polynomial degrees. Betas and powers are 
	dicts with keys as poly power (1,2,3 ...) as created by the script. Beta values are the beta values (duh) and powers an array of (x,y) powers for
	the different parameters. 
	"""
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(nrows=1, ncols=len(degrees), figsize=(14,5), gridspec_kw={'width_ratios': degrees})

	axes[0].set_ylabel(r"$\beta$", fontsize=14)
	# axes[1].set_xlabel(r"Coef of $x^i y^j$", fontsize=14)
	fig.supxlabel(r"Coef of $x^i y^j$")

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
		ax.set_xticks(ticks, labels, rotation=90)
		ax.plot(ticks, betas[p], c="k", ls="--", marker=".")
		ax.fill_between(ticks, (betas[p]-2*betas_se[p]), (betas[p]+2*betas_se[p]), color='r', alpha=.3)

	fig.tight_layout()
	if filename is not None: plt.savefig(utils.make_figs_path(filename), dpi=300)
	plt.show()


def solve_a(n=1000, train_size=0.8, noise_std=0.1, random_state=123):
	"""
	Function to preform what I assume task b is asking. R2 and MSE (test and train) for polynomials (1,5)

	1. Generate x, y data from Franke Function 
	2. Scale design matrix per column  
	3. Make polynomial features using the x and y values from design matrix
	4. Calculate measures and save coefs
	5. Plot mse & r2
	6. Plot chaos beta plot
	"""
	D = make_FrankeFunction(n=n, uniform=True, random_state=random_state, noise_std=noise_std)

	degrees = np.arange(1, 12+1)
	mse_train = np.zeros_like(degrees, dtype=float)
	mse_test = np.zeros_like(degrees, dtype=float)
	r2_train = np.zeros_like(degrees, dtype=float)
	r2_test = np.zeros_like(degrees, dtype=float)
	betas = {}
	betas_se = {}
	powers = {} 

	for i, degree in enumerate(degrees):
		Dp, powers[degree] = D.polynomial(degree=degree, return_powers=True)
		Dp = Dp.scaled(scheme="Standard")

		Dp_train, Dp_test = Dp.train_test_split(ratio=train_size, random_state=random_state)
		reg = LinearRegression(method="pINV").fit(Dp_train)

		mse_train[i] = reg.mse(Dp_train)
		mse_test[i] = reg.mse(Dp_test)

		r2_train[i] = reg.r2_score(Dp_train)
		r2_test[i] = reg.r2_score(Dp_test)

		betas[degree] = reg.coef_
		var_beta =	 np.diag(reg.coef_var(Dp_train.X, noise_std))
		betas_se[degree] = 2*np.sqrt(var_beta)

	# Show mse and r2 score 

	return (degrees, mse_train, mse_test, r2_train, r2_test), (betas, betas_se, powers)	

if __name__ == "__main__":
	params1, params2 = solve_a(n=600, noise_std=0.1, random_state=321, train_size=2/3)

	# plot_mse_and_r2(*params1, filename="mse_and_r2_linreg")
	plot_beta_progression(*params2, filename="linreg_coefs")