'''
This script intends to make analysis plots comparing different Gradient Descent
methods.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
from time import time
import plot_utils

import context
from sknotlearn.data import Data
from sknotlearn.linear_model import OLS_gradient, ridge_gradient
from sknotlearn.optimize import GradientDescent, SGradientDescent


def tune_learning_rate(
    data_train: Data,
    data_val: Data,
    theta0: np.ndarray,
    learning_rates: np.ndarray,
    optimizers: tuple,
    optimizer_names: tuple,
    lmbda: float = None,
    ylims: tuple = (0, 3),
    filename: str = None,
    verbose: bool = False
) -> None:
    """Plots the MSE score on validation data of the minimisation of either OLS
    or Ridge cost functions using given GradientDescent instances.

    Args:
        data_train (Data): Data to optimise linear regression.
        data_val (Data): Validation data to apply linear regression on.
        theta0 (np.ndarray): Initial parameters for descent. If list or tuple
                             starting points are iterated over.
        learning_rates (np.ndarray): Array of learning rates to use.
        optimizers (tuple): Iterable of instances of GradientDescent to plot.
        optimizer_names (tuple): Legend names for the GradientDescent methods.
        lmbda (float, optional): In case of "ridge" cost function requiring a
                                 lmbda hyperparameter. Defaults to None.
        ylims (tuple, optional): Limits for MSE score in plot.
                                 Defaults to (0,3).
        filename (str, optional): Filename to save figure as. Defaults to None.
        verbose (bool, optional): Whether to print best MSE-scores of each
                                  optimizer. Defaults to False.
    """
    # Wrapping theta0 in tuple if 1 starting point is given.
    if not isinstance(theta0, (tuple, list)):
        theta0 = (theta0,)

    # Unpacking data
    y_train, X_train = data_train.unpacked()
    y_val, X_val = data_val.unpacked()

    # Set correct gradient to use
    if lmbda is None:
        def grad(theta, idcs): return OLS_gradient(theta,
                                                   data_train,
                                                   idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train) @ X_train.T @ y_train
    else:
        def grad(theta, idcs): ridge_gradient(theta,
                                              data_train,
                                              lmbda,
                                              idcs)
        theta_ana = np.linalg.pinv(X_train.T@X_train +
                                   lmbda*np.eye(X_train.shape[1])) @ X_train.T @ y_train

    # Iterate through optimizers and compute/plot scores
    for optimizer, name, color in zip(optimizers,
                                      optimizer_names,
                                      plot_utils.colors[:len(optimizers)]):
        start_time = time()     # For timing algorithms
        MSE_array = np.zeros((len(theta0), len(learning_rates)))
        for i, x0 in enumerate(theta0):
            for j, learning_rate in enumerate(learning_rates):
                # Setting the correct learning rate
                params = optimizer.params
                params["eta"] = learning_rate
                optimizer.set_params(params)

                # The call-signature of GD and SGD is slightly different
                if isinstance(optimizer, SGradientDescent):
                    theta_opt = optimizer.call(
                        grad=grad,
                        x0=x0,
                        all_idcs=np.arange(len(data_train))
                    )
                else:
                    theta_opt = optimizer.call(
                        grad=grad,
                        x0=x0,
                        args=(np.arange(len(data_train)),)
                    )
                if optimizer.converged:
                    MSE_array[i, j] = np.mean((X_val@theta_opt - y_val)**2)
                else:
                    MSE_array[i, j] = np.nan
        # Average MSE across theta0s
        MSE_means = MSE_array.mean(axis=0)
        plt.plot(learning_rates, MSE_means, label=name, c=color)
        if len(theta0) > 1:     # Adding 95% confidence interval
            MSE_stds = MSE_array.std(axis=0) / np.sqrt(len(theta0)-1)
            plt.fill_between(learning_rates,
                             MSE_means-2*MSE_stds, MSE_means+2*MSE_stds,
                             alpha=0.3, color=color)

        argbest = np.nanargmin(MSE_means)
        plt.annotate("",
                     xy=(learning_rates[argbest], MSE_means[argbest]),
                     xytext=(learning_rates[argbest],
                             MSE_means[argbest] - ylims[1]/6),
                     ha="center",
                     arrowprops=dict(facecolor=color))
        if verbose:
            print(f"{name} MSE score: {MSE_means[argbest]:.4} "
                  f"+- {MSE_stds[argbest]:.3} "
                  f"(Learning rate {learning_rates[argbest]:.2}) "
                  f"({time()-start_time:.2f} s)")

    # Calculating analytical solution from matrix inversion
    MSE_ana = np.mean((X_val@theta_ana - y_val)**2)
    if verbose:
        print(f"Analytical MSE score: {MSE_ana:.4}")

    # Plotting analytical solution as grey line
    if lmbda is None:
        hlabel = "Analytical OLS solution"
    else:
        hlabel = "Analytical ridge solution"
    plt.hlines(MSE_ana,
               xmin=learning_rates.min(),
               xmax=learning_rates.max(),
               label=hlabel,
               ls="--",
               colors="grey"
               )
    plt.ylim(ylims)
    plt.xlabel("Learning rate")
    plt.ylabel("Validation MSE")
    plt.legend(
        bbox_to_anchor=(0, 1.13, 1, 0.2),
        loc="upper left",
        ncol=2,
        mode="expand"
    )
    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def tune_lambda_learning_rate(
    data_train: Data,
    data_val: Data,
    theta0: np.ndarray,
    learning_rates: np.ndarray,
    lmbdas: np.ndarray,
    optimizer: GradientDescent,
    vlims: tuple = (0, 3),
    title: str = None,
    filename: str = None,
    verbose: bool = False
) -> None:
    """Plots heatmap of validation MSE scores of minimization of Ridge cost
    function using given GradientDescent instance.

    Args:
        data (Data): Data to train on. Is train-test split.
        trainsize (float): The ratio of data to be used for training.
        learning_rates (np.ndarray): Array of learning rates to use.
        lmbdas (np.ndarray): Array of lambda-values to use for cost function.
        optimizer (GradientDescent): GradientDescent instance to use
                                     for optimization.
        vlims (tuple, optional): Limits for MSE colorbar. Defaults to (0,3).
        filename (str, optional): Filename to save figure as. Defaults to None.
        verbose (bool, optional): Whether to print best MSE-scores of each
                                  optimizer. Defaults to False.
    """
    # Wrapping theta0 in tuple if 1 starting point is given.
    if not isinstance(theta0, (tuple, list)):
        theta0 = (theta0,)

    # Unpacking data
    y_val, X_val = data_val.unpacked()

    # Iterate through every combination of lambda and learning rate.
    MSE_grid = np.zeros((len(theta0), len(lmbdas), len(learning_rates)))
    start_time = time()  # For timing the algorithm
    for i, x0 in enumerate(theta0):
        for j, lmbda in enumerate(lmbdas):
            # Set gradient function
            def grad(theta, idcs): return ridge_gradient(theta,
                                                         data_train,
                                                         lmbda,
                                                         idcs)
            for k, learning_rate in enumerate(learning_rates):
                # Set learning rate
                params = optimizer.params
                params["eta"] = learning_rate
                optimizer.set_params(params)

                # Call signature is slightly different between GD and SGD.
                if isinstance(optimizer, SGradientDescent):
                    theta_opt = optimizer.call(
                        grad=grad,
                        x0=x0,
                        all_idcs=np.arange(len(data_train))
                    )
                else:
                    theta_opt = optimizer.call(
                        grad=grad,
                        x0=x0,
                        args=(np.arange(len(data_train)),)
                    )
                if optimizer.converged:
                    MSE_grid[i, j, k] = np.mean((X_val@theta_opt-y_val)**2)
                else:
                    MSE_grid[i, j, k] = np.nan
    # Average MSE across theta0s
    mean_MSE_grid = MSE_grid.mean(axis=0)
    # Finding optimum
    arg_best_MSE = np.unravel_index(
        np.nanargmin(mean_MSE_grid),
        np.shape(mean_MSE_grid)
    )
    if verbose:
        print(f"Best MSE value of {optimizer.method}: "
              f"{mean_MSE_grid[arg_best_MSE]:.4} "
              f"with lmbda {lmbdas[arg_best_MSE[0]]:.1E} "
              f"lrate {learning_rates[arg_best_MSE[1]]:.2} "
              f"({time()-start_time:.2f} s)")

    fig, ax = plt.subplots()
    # Plot heatmap first
    sns.heatmap(
        mean_MSE_grid,
        vmin=vlims[0], vmax=vlims[1],
        annot=True,
        cmap=plot_utils.cmap,
        ax=ax,
        cbar_kws={'label': 'Validation MSE'}
    )
    # Plot a red triangle around best value. This code is messi (not ronaldo)
    ax.add_patch(plt.Rectangle((arg_best_MSE[1],
                                arg_best_MSE[0]),
                               1, 1,
                               fc='none',
                               ec='red',
                               lw=5,
                               clip_on=False))

    # Set xylabels and ticks. This code is ronaldo, but it works.
    ax.set_xlabel("Learning rate")
    ax.set_xticks(
        np.arange(len(learning_rates))[::2],
        labels=[f"{lrate:.4g}" for lrate in learning_rates][::2]
    )
    ax.set_ylabel(r"$\log_{10}(\lambda)$")
    ax.set_yticks(
        np.arange(len(lmbdas))[::2],
        labels=np.log10(lmbdas)[::2]
    )
    if title is not None:
        ax.set_title(title)

    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


if __name__ == "__main__":
    # Import data
    from sknotlearn.datasets import make_FrankeFunction, load_Terrain
    # D = load_Terrain(n=600, random_state=321)
    D = make_FrankeFunction(n=600, noise_std=0.1, random_state=321)
    D = D.polynomial(degree=5, with_intercept=False)
    D_train, D_val = D.train_test_split(ratio=3/4, random_state=42)
    D_train = D_train.scaled(scheme="Standard")
    D_val = D_train.scale(D_val)

    # Setting some general params
    random_state = 123
    np.random.seed(random_state)
    theta0 = [np.random.randn(D_val.n_features) for _ in range(5)]

    max_iter = 100

    batch_size = 2**6

    # All the gradient descent instances
    GD = GradientDescent("plain", dict(eta=0.),
                         its=max_iter)
    mGD = GradientDescent("momentum", dict(gamma=0.8, eta=0.),
                          its=max_iter)
    SGD = SGradientDescent("plain", dict(eta=0.),
                           epochs=max_iter, batch_size=batch_size,
                           random_state=random_state)
    mSGD = SGradientDescent("momentum", dict(gamma=0.8, eta=0.),
                            epochs=max_iter, batch_size=batch_size,
                            random_state=random_state)
    mSGD2 = SGradientDescent("momentum", dict(gamma=0.01, eta=0.),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    mSGD3 = SGradientDescent("momentum", dict(gamma=0.1, eta=0.),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    mSGD4 = SGradientDescent("momentum", dict(gamma=0.5, eta=0.),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    mSGD5 = SGradientDescent("momentum", dict(gamma=1., eta=0.),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    SGDb = SGradientDescent("plain", dict(eta=0.),
                            epochs=max_iter, batch_size=batch_size//4,
                            random_state=random_state)
    SGDe = SGradientDescent("plain", dict(eta=0.),
                            epochs=max_iter*4, batch_size=batch_size,
                            random_state=random_state)
    SGDbe = SGradientDescent("plain", dict(eta=0.),
                             epochs=max_iter*4, batch_size=batch_size//4,
                             random_state=random_state)
    aGD = GradientDescent("adagrad", dict(eta=0.),
                          its=max_iter)
    maGD = GradientDescent("adagrad_momentum", dict(gamma=0.8, eta=0.),
                           its=max_iter)
    aSGD = SGradientDescent("adagrad", dict(eta=0.),
                            epochs=max_iter, batch_size=batch_size,
                            random_state=random_state)
    maSGD = SGradientDescent("adagrad_momentum", dict(gamma=0.8, eta=0.),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    rmSGD = SGradientDescent("rmsprop", dict(eta=0., beta=0.9),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    adSGD = SGradientDescent("adam", dict(eta=0., beta1=0.9, beta2=0.99),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)

    #####################
    # lmbda plot params #
    #####################
    params1 = {
        "PGD_MGD_PSGD_MSGD": dict(
            learning_rates=np.linspace(0., 0.14, 101)[1:],
            optimizers=(GD, mGD, SGD, mSGD),
            optimizer_names=(f"Plain GD", f"Momentum GD",
                             "Plain SGD", "Momentum SGD"),
            ylims=(0, 0.8)
        ),
        "SGD_batches_epochs": dict(
            learning_rates=np.linspace(0., 0.08, 101)[1:],
            optimizers=(SGD, SGDb, SGDe, SGDbe),
            optimizer_names=(f"SGD", fr"SGD, $4\times$batches",
                             fr"SGD, $4\times$epochs",
                             fr"SGD, $4\times$batches/epochs"),
            ylims=(0, 0.6)
        ),
        "adagrad": dict(
            learning_rates=np.linspace(0., 0.7, 101)[1:],
            optimizers=(aGD, maGD, aSGD, maSGD),
            optimizer_names=(f"AdaGrad GD", f"AdaGradMom GD",
                             "AdaGrad SGD", "AdaGradMom SGD"),
            ylims=(0, 0.8)
        ),
        "momentum": dict(
            learning_rates=np.linspace(0., 0.08, 101)[1:],
            optimizers=(mSGD2, mSGD3, mSGD4, mSGD, mSGD5),
            optimizer_names=(r"mSGD $\gamma=0.01$", r"mSGD $\gamma=0.1$",
                             r"mSGD $\gamma=0.5$", r"mSGD $\gamma=0.8$",
                             r"mSGD $\gamma=1$"),
            ylims=(0, 0.8)
        ),
        "tunable": dict(
            # funky learning rate to get more small eta evaluations
            learning_rates=np.linspace(0.001**(1/3), 0.7**(1/3), 101)**(3),
            optimizers=(rmSGD, adSGD, aSGD, maSGD),
            optimizer_names=("RMSprop SGD", "Adam SGD",
                             "AdaGrad SGD", "AdaGradMom SGD"),
            ylims=(0, 0.8)
        ),
        "test2": dict(
            learning_rates=np.linspace(0.0001, 0.1, 101),
            optimizers=(adSGD,),
            optimizer_names=("Adam",),
            ylims=(0, 0.8)
        )
    }
    #######################
    # heatmap plot params #
    #######################
    params2 = {
        "plain_GD": dict(
            learning_rates=np.linspace(0., 0.08, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=GD,
            vlims=(None, 0.4),
        ),
        "momentum_GD": dict(
            learning_rates=np.linspace(0., 0.14, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=mGD,
            vlims=(None, 0.6),
            # title="GD with momentum"
        ),
        "plain_SGD": dict(
            learning_rates=np.linspace(0., 0.05, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=SGD,
            vlims=(None, 0.4)
        ),
        "momentum_SGD": dict(
            learning_rates=np.linspace(0., 0.11, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=mSGD,
            vlims=(None, 0.6),
            # title="SGD with momentum"
        ),
        "adagrad_SGD": dict(
            learning_rates=np.linspace(0., 1., 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=aSGD,
            vlims=(None, 0.4)
        ),
        "adagrad_momentum_SGD": dict(
            learning_rates=np.linspace(0., 0.6, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=maSGD,
            vlims=(None, None),
            # title="SGD AdaGrad with momentum"
        ),
        "adam_SGD": dict(
            learning_rates=np.linspace(0., 0.1, 11)[1:],
            lmbdas=np.logspace(-5, 1, 13),
            optimizer=adSGD,
            vlims=(None, 0.7),
            # title="SGD Adam with momentum"
        )
    }

    # Choosing plot to plot
    plot1 = "adagrad"
    # plot1 = None
    # plot2 = "adagrad_momentum_SGD"
    plot2 = None

    # Plotting
    if plot1:
        tune_learning_rate(
            data_train=D_train,
            data_val=D_val,
            theta0=theta0,
            verbose=True,
            **params1[plot1],
            # filename="learning_rates_"+plot1
        )

    if plot2:
        tune_lambda_learning_rate(
            data_train=D_train,
            data_val=D_val,
            theta0=theta0,
            verbose=True,
            **params2[plot2],
            # filename="lmbda_learning_rates_"+plot2
        )
