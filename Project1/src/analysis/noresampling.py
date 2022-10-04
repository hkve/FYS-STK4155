import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Custom stuff
import context
from sknotlearn.linear_model import LinearRegression, Ridge, Lasso
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data
from utils import make_figs_path, colors


def make_power_labels(p, powers):
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

	return labels


def plot_train_test(degrees, train, test, filename = None, **kwargs):
	"""
	Function to plot train/test mse and r2. Simple line plot with mse on the left and r2 on the right
	"""
	opt = { # Plotting options
		"xlabel": r"Polynomial degree",
		"ylabel": r"MSE",
		"fontsize": 14,
		"title": "OLS no resampling",
		"title_fontsize": 16
	}

	opt.update(kwargs)

	sns.set_style("darkgrid")
	fig, ax = plt.subplots()
	ax.plot(degrees, train, c=colors[0], label="Train", alpha=0.75, lw=2.5)
	ax.plot(degrees, test, c=colors[1], label="Test", lw=2.5)
	ax.set_title(opt["title"], fontsize=opt["title_fontsize"])
	ax.set_xlabel(opt["xlabel"], fontsize=opt["fontsize"])
	ax.set_ylabel(opt["ylabel"], fontsize=opt["fontsize"])
	ax.legend(fontsize=opt["fontsize"])

	fig.tight_layout()
	if filename is not None: plt.savefig(make_figs_path(filename), dpi=300)
	plt.show()


def plot_theta_progression(thetas, thetas_se, powers, degrees=[1,3,5], filename=None):
	"""
	Function to plot the evolution of of fitted paramters based on different polynomial degrees. Betas and powers are 
	dicts with keys as poly power (1,2,3 ...) as created by the script. Beta values are the theta values (duh) and powers an array of (x,y) powers for
	the different parameters. 
	"""
	sns.set_style("darkgrid")
	fig, axes = plt.subplots(nrows=1, ncols=len(degrees), figsize=(14,5), gridspec_kw={'width_ratios': degrees})

	axes[0].set_ylabel(r"$\theta$", fontsize=14)
	fig.suptitle("OLS coefficients", fontsize=16)
	fig.supxlabel(r"$\theta$ corresponding to $x^i y^j$", fontsize=14)

	x = [0,0,60]
	for ax, p, x_ in zip(axes, degrees, x):
		# Make labels like 1, x, y, xy ...
		labels = make_power_labels(p, powers)
		
		# Plot for degree p
		ticks = np.arange(len(labels))
		ax.set_xticks(ticks, labels, rotation=x_)
		ax.plot(ticks, thetas[p], c=colors[0], ls="--", marker=".", lw=2.5)
		ax.fill_between(ticks, (thetas[p]-2*thetas_se[p]), (thetas[p]+2*thetas_se[p]), color=colors[0], alpha=.3)

	fig.tight_layout()
	if filename is not None: plt.savefig(make_figs_path(filename), dpi=300)
	plt.show()


def plot_theta_heatmap(thetas, theta_se, powers, degrees=[1,2,3,4,5], filename=None):
	"""
	Shows table like view of theta coefs. This shows the absoulte value of theta since log requires them to be positive.

	To use darkgrid (which looks strange), comment/uncomment as instructed
	"""
	h, w = len(degrees), len(thetas[degrees[-1]])
	thetas_mat = np.zeros((h, w))


	for i, p in enumerate(degrees):
		labels = make_power_labels(p, powers)
		w_p = len(thetas[p])

		thetas_mat[i, :w_p] = thetas[p]

	thetas_mat_ = abs(thetas_mat)
	# thetas_mat = np.where(thetas_mat < 1e-2, np.nan, thetas_mat)

	sns.set_style("white") # and comment out this
	fig, ax = plt.subplots(figsize=(12,5))
	
	im = ax.imshow(thetas_mat_, cmap="viridis", norm=LogNorm(vmin=np.nanmin(thetas_mat_)+1e-2, vmax=np.nanmax(thetas_mat_)))

	# Show all ticks and label them with the respective list entries
	ax.set_xticks(np.arange(len(labels)), labels=labels)
	ax.set_yticks(np.arange(len(degrees)), labels=degrees)
	cbar = fig.colorbar(im, pad=0.01, shrink=0.55, aspect=6)
	cbar.set_label(r"$|\theta|$", fontsize=14)
	ax.set_xlabel(r"$\theta$ corresponding to $x^i y^j$", fontsize=14)
	ax.set_ylabel(r"Polynomial degree", fontsize=14)
	ax.set_title("OLS coefficients", fontsize=16)

	for i in range(len(degrees)):
		for j in range(len(labels)):
			value = thetas_mat[i, j]
			# if np.isnan(value):
			# 	continue
			ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="w")
	
	fig.tight_layout()
	if filename is not None: plt.savefig(make_figs_path(filename), dpi=300, bbox_inches='tight')
	plt.show()


def run_no_resampling(Model, degrees, n=1000, train_size=0.8, noise_std=0.1, random_state=123, lmbda=None):
	"""
	Function to preform what I assume task b is asking. R2 and MSE (test and train) for polynomials (1,5)

	1. Generate x, y data from Franke Function 
	2. Scale design matrix per column  
	3. Make polynomial features using the x and y values from design matrix
	4. Calculate measures and save coefs
	5. Plot mse & r2
	6. Plot chaos theta plot
	"""
	D = make_FrankeFunction(n=n, uniform=True, random_state=random_state, noise_std=noise_std)

	mse_train = np.zeros_like(degrees, dtype=float)
	mse_test = np.zeros_like(degrees, dtype=float)
	r2_train = np.zeros_like(degrees, dtype=float)
	r2_test = np.zeros_like(degrees, dtype=float)
	thetas = {}
	thetas_se = {}
	powers = {} 

	for i, degree in enumerate(degrees):
		Dp, powers[degree] = D.polynomial(degree=degree, return_powers=True)
		Dp_train, Dp_test = Dp.train_test_split(ratio=train_size, random_state=random_state)

		Dp_train = Dp_train.scaled(scheme="Standard")
		Dp_test = Dp_train.scale(Dp_test)
		
		if Model in [Ridge, Lasso]:
			assert lmbda is not None, f"When using {Model = }, please pass a lambda value"
			reg = Model(lmbda=lmbda).fit(Dp_train)
		else:
			reg = Model().fit(Dp_train)
		
		mse_train[i] = reg.mse(Dp_train)
		mse_test[i] = reg.mse(Dp_test)

		r2_train[i] = reg.r2_score(Dp_train)
		r2_test[i] = reg.r2_score(Dp_test)

		thetas[degree] = reg.coef_
		var_theta =	 np.diag(reg.coef_var(Dp_train.X, noise_std))
		thetas_se[degree] = 2*np.sqrt(var_theta)

	# Show mse and r2 score 
	print(f"Found best degree p = {degrees[np.argmin(mse_test)]} with MSE = {np.min(mse_test)} for {Model.__name__}")
	return (mse_train, mse_test, r2_train, r2_test), (thetas, thetas_se, powers)	

if __name__ == "__main__":
	# Set the number of degrees you want to try:
	degrees = np.arange(1, 12+1)
	
	# Slicing [:3] is just a very hacky way if you only want to plot some
	Models = [LinearRegression, Ridge, Lasso][:3]
	lmbdas = [None, 1e-3, 1e-3][:3]
	names = ["OLS", "Ridge", "Lasso"][:3]

	for Model, lmbda, name in zip(Models, lmbdas, names):
		params1, params2 = run_no_resampling(Model, degrees,n=600, noise_std=0.1, random_state=321, train_size=2/3, lmbda=lmbda)
		mse_train, mse_test, r2_train, r2_test = params1

		title = f"{name} no resampling"
		plot_train_test(degrees, mse_train, mse_test, filename=f"{name}_mse_noresample", title=title)
		plot_train_test(degrees, r2_train, r2_test, filename=f"{name}_R2_noresample", ylabel=r"R$^2$-score", title=title)

		if Model == LinearRegression:
			plot_theta_progression(*params2, filename="OLS_coefs_plots")
			plot_theta_heatmap(*params2, filename="OLS_coefs_table")