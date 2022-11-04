import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
import plot_utils
from utils import make_figs_path

import context
from sknotlearn.data import Data
from sknotlearn.linear_model import OLS_gradient, ridge_gradient
from sknotlearn.optimize import GradientDescent, SGradientDescent

def tune_learning_rate(
    data:Data,
    trainsize:float,
    learning_rates:np.ndarray,
    optimizers:tuple,
    optimizer_names:tuple,
    cost:str = "OLS",
    lmbda:float=None,
    ylims:tuple = (0,3),
    filename:str = None,
    random_state:int = None,
    verbose:bool = False
) -> None:
    """Plots the MSE score on validation data of the minimisation of either OLS or Ridge cost functions using given GradientDescent instances.

    Args:
        data (Data): Data to train on. Is train-test split.
        trainsize (float): The ratio of data to be used for training.
        learning_rates (np.ndarray): Array of learning rates to use.
        optimizers (tuple): Iterable of instances of GradientDescent to plot for.
        optimizer_names (tuple): Legend names for the GradientDescent methods.
        cost (str, optional): Cost function to use. Etiher "OLS" or "ridge". Defaults to "OLS".
        lmbda (float, optional): In case of "ridge" cost function requiring a lmbda hyperparameter. Defaults to None.
        ylims (tuple, optional): Limits for MSE score in plot. Defaults to (0,3).
        filename (str, optional): Filename to save figure as. Defaults to None.
        random_state (int, optional): np.random seed to use for rng. Defaults to None.
        verbose (bool, optional): Whether to print best MSE-scores of each optimizer. Defaults to False.
    """

    # Split data into training and validation data
    data_train, data_val = data.train_test_split(ratio=trainsize, random_state=random_state)
    y_train, X_train = data_train.unpacked()
    y_val, X_val = data_val.unpacked()

    # Set random seed
    if random_state:
        np.random.seed(random_state)
    # Set starting point for of minimisation
    theta0 = np.random.randn(X_val.shape[1])

    # Set correct gradient to use
    if cost == "OLS":
        grad = lambda theta, idcs : OLS_gradient(theta, data_train, idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train) @ X_train.T @ y_train
    elif cost == "ridge":
        grad = lambda theta, idcs : ridge_gradient(theta, data_train, lmbda, idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train + lmbda*np.eye(X_train.shape[1])) @ X_train.T @ y_train

    # Iterate through optimizers and compute/plot scores
    for optimizer, name in zip(optimizers, optimizer_names):
        MSE_list = list()
        for learning_rate in learning_rates:
            # Setting the correct learning rate
            params = optimizer.params
            params["eta"] = learning_rate
            optimizer.set_params(params)

            # The call-signature of GD and SGD is slightly different
            if isinstance(optimizer, SGradientDescent):
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    all_idcs=np.arange(len(data_train))
                )
            else:
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    args=(np.arange(len(data_train)),)
                )

            
            MSE_list.append(np.mean((X_val@theta_opt - y_val)**2))
        if verbose:
            print(f"{name} MSE score: {np.nanmin(MSE_list)} (Learning rate {learning_rates[np.nanargmin(MSE_list)]})")
        plt.plot(learning_rates, MSE_list, label=name)

    # Calculating analytical solution from matrix inversion
    MSE_ana = np.mean((X_val@theta_ana - y_val)**2)
    if verbose:
        print(f"Analytical MSE score: {MSE_ana}")

    # Plotting analytical solution as grey line
    plt.hlines(MSE_ana,
        xmin=learning_rates.min(),
        xmax=learning_rates.max(),
        label=f"Analytical {cost} solution",
        ls="--",
        colors="grey"
    )
    plt.ylim(ylims)
    plt.xlabel("Learning rate")
    plt.ylabel("Validation MSE")
    plt.legend(bbox_to_anchor=(0, 1.13, 1, 0.2), loc="upper left", ncol=2, mode="expand")
    if filename:
        plt.savefig(make_figs_path(filename))
    plt.show()


def tune_lambda_learning_rate(
    data:Data,
    trainsize:float,
    learning_rates:np.ndarray,
    lmbdas:np.ndarray,
    optimizer:GradientDescent,
    ylims:tuple = (0,3),
    filename:str = None,
    random_state:int = None,
    verbose:bool = False
) -> None:
    """Plots heatmap of validation MSE scores of minimization of Ridge cost function using given GradientDescent instance.

    Args:
        data (Data): Data to train on. Is train-test split.
        trainsize (float): The ratio of data to be used for training.
        learning_rates (np.ndarray): Array of learning rates to use.
        lmbdas (np.ndarray): Array of lambda-values to use for cost function.
        optimizer (GradientDescent): GradientDescent instance to use for optimization.
        vlims (tuple, optional): Limits for MSE colorbar. Defaults to (0,3).
        filename (str, optional): Filename to save figure as. Defaults to None.
        random_state (int, optional): np.random seed to use for rng. Defaults to None.
        verbose (bool, optional): Whether to print best MSE-scores of each optimizer. Defaults to False.
    """

    # Split data into training and validation data
    data_train, data_val = data.train_test_split(ratio=trainsize, random_state=random_state)
    y_train, X_train = data_train.unpacked()
    y_val, X_val = data_val.unpacked()

    # Set random seed
    if random_state:
        np.random.seed(random_state)
    # Set starting point for of minimisation
    theta0 = np.random.randn(X_val.shape[1])

    # Iterate through every combination of lambda and learning rate.
    MSE_grid = list()
    for lmbda in lmbdas:
        MSE_list = list()
        # Set gradient
        grad = lambda theta, idcs : ridge_gradient(theta, data_train, lmbda, idcs)
        for learning_rate in learning_rates:
            # Set learning rate
            params = optimizer.params
            params["eta"] = learning_rate
            optimizer.set_params(params)

            # Call signature is slightly different between GD and SGD.
            if isinstance(optimizer, SGradientDescent):
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    all_idcs=np.arange(len(data_train))
                )
            else:
                theta_opt = optimizer.call(
                    grad=grad,
                    x0=theta0,
                    args=(np.arange(len(data_train)),)
                )
            MSE_list.append(np.mean((X_val@theta_opt-y_val)**2))
        MSE_grid.append(MSE_list)
    
    fig, ax = plt.subplots()
    # Plot heatmap first
    sns.heatmap(MSE_grid, vmin=ylims[0], vmax=ylims[1], annot=True, cmap="viridis", ax=ax)
    # Plot a red triangle around best value. This code is messi (not ronaldo)
    arg_best_MSE = np.unravel_index(np.argmin(MSE_grid), np.shape(MSE_grid))
    ax.add_patch(plt.Rectangle((arg_best_MSE[1], arg_best_MSE[0]), 1, 1, fc='none', ec='red', lw=5, clip_on=False))

    # Set xylabels and ticks. This code is ronaldo, but it works.
    ax.set_xlabel("Learning rate")
    ax.set_xticks(
        np.arange(len(learning_rates))[::2],
        labels=[f"{lrate:.4g}" for lrate in learning_rates][::2]
    )
    ax.set_ylabel(r"$\log_{10}(\lambda)$")
    ax.set_yticks(
        np.arange(len(lmbdas)),
        labels=np.log10(lmbdas)
    )

    if filename:
        plt.savefig(make_figs_path(filename))
    plt.show()

    
if __name__=="__main__":
    from sknotlearn.datasets import make_FrankeFunction
    random_state = 321
    D = make_FrankeFunction(n=600, noise_std=0.1, random_state=random_state)
    D = D.polynomial(degree=5).scaled(scheme="Standard")
    D.X = D.X[:,1:]

    max_iter = 100

    n_batches = 32
    batch_size = (3 * len(D) // 4) // n_batches
    # batch_size = 32

    # GD = GradientDescent("plain", {"eta":0.}, its=max_iter)
    # mGD = GradientDescent("momentum", {"gamma":0.8, "eta":0.}, its=max_iter)
    SGD = SGradientDescent("plain", {"eta":0.}, epochs=max_iter, batch_size=batch_size, random_state=random_state)
    # mSGD = SGradientDescent("momentum", {"gamma":0.8, "eta":0.}, epochs=max_iter, batch_size=batch_size, random_state=random_state)
    SGDb = SGradientDescent("plain", {"eta":0.}, epochs=max_iter, batch_size=batch_size//4, random_state=random_state)
    SGDe = SGradientDescent("plain", {"eta":0.}, epochs=4*max_iter, batch_size=batch_size, random_state=random_state)
    SGDbe = SGradientDescent("plain", {"eta":0.}, epochs=4*max_iter, batch_size=batch_size//4, random_state=random_state)
    # aGD = GradientDescent("adagrad", {"eta":0.}, its=max_iter)
    # maGD = GradientDescent("adagrad_momentum", {"gamma":0.8, "eta":0.}, its=max_iter)
    # aSGD = SGradientDescent("adagrad", {"eta":0.}, epochs=max_iter, batch_size=batch_size, random_state=random_state)
    # maSGD = SGradientDescent("adagrad_momentum", {"gamma":0.8, "eta":0.}, epochs=max_iter, batch_size=batch_size, random_state=random_state)
    # rmSGD = SGradientDescent("rmsprop", {"eta":0., "beta":0.9}, epochs=max_iter, batch_size=batch_size, random_state=random_state)
    # adSGD = SGradientDescent("adam", {"eta":0., "beta1":0.9, "beta2":0.9}, epochs=max_iter, batch_size=batch_size, random_state=random_state)


    # learning_rates = np.linspace(0.001, 0.13, 101)
    learning_rates = np.linspace(0.001, 0.05, 101)
    # learning_rates = np.linspace(0.001, 0.6, 101)
    # learning_rates = np.linspace(0.001, 0.13, 101)
    tune_learning_rate(
        data=D,
        trainsize=3/4,
        learning_rates=learning_rates,
        # optimizers=(GD,mGD,SGD,mSGD),
        optimizers=(SGD,SGDb,SGDe,SGDbe),
        # optimizers=(aGD,maGD,aSGD,maSGD),
        # optimizers=(rmSGD,adSGD),
        # optimizer_names=(f"Plain GD",f"Momentum GD", "Plain SGD", "Momentum SGD"),
        optimizer_names=(f"SGD",fr"SGD, $4\times$batches", fr"SGD, $4\times$epochs", fr"SGD, $4\times$batches/epcochs"),
        # optimizer_names=(f"AdaGrad GD",f"AdaGradMom GD", "AdaGrad SGD", "AdaGradMom SGD"),
        # optimizer_names=("RMSprop","Adam"),
        cost="OLS",
        ylims=(0,0.8),
        verbose=True,
        random_state=random_state,
        # filename="learning_rates_PGD_MGD_PSGD_MSGD"
        filename="learning_rates_SGD_batches_epochs"
        # filename="learning_rates_adagrad"
    )

    # tune_lambda_learning_rate(
    #     data=D,
    #     trainsize=3/4,
    #     learning_rates=np.linspace(0.001, 0.08, 11, endpoint=True),
    #     lmbdas=np.logspace(-11,-1,11),
    #     optimizer=SGD,
    #     ylims=(None,0.4),
    #     random_state=random_state
    # )