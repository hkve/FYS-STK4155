"""This file aims to make analysis plots of gradient descent methods
used in optimising OLS and ridge cost functions of linear models"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
from time import time

from gradient_descent_utils import *
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
    ylims: tuple = (None, None),
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
        theta_ana = OLS_solution(X_train, y_train)
    else:
        def grad(theta, idcs): return ridge_gradient(theta,
                                                     data_train,
                                                     lmbda,
                                                     idcs)
        theta_ana = ridge_solution(X_train, y_train, lmbda)

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

                theta_opt = call_optimizer(optimizer,
                                           grad, x0,
                                           np.arange(len(data_train)))
                # Check if gradient exploded
                MSE_array[i, j] = clip_exploded_gradients(
                    val=MSE(X_val@theta_opt, y_val),
                    lim=10*ylims[1] if ylims is not None else None,
                    convergence=optimizer.converged
                )
        # Average MSE across theta0s
        MSE_means = MSE_array.mean(axis=0)
        plt.plot(learning_rates, MSE_means, label=name, c=color)

        if len(theta0) > 1:  # Adding 95% confidence interval
            MSE_stds = MSE_array.std(axis=0) / np.sqrt(len(theta0)-1)
            plt.fill_between(learning_rates,
                             MSE_means-2*MSE_stds, MSE_means+2*MSE_stds,
                             alpha=0.3, color=color)

        # Annotating the best MSE value
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
                  f"(Learning rate {learning_rates[argbest]}) "
                  f"({time()-start_time:.2f} s)")

    # Calculating analytical solution from matrix inversion
    MSE_ana = MSE(X_val@theta_ana, y_val)
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

                theta_opt = call_optimizer(optimizer,
                                           grad, x0,
                                           np.arange(len(data_train)))
                MSE_grid[i, j, k] = clip_exploded_gradients(
                    val=MSE(X_val@theta_opt, y_val),
                    lim=10*vlims[1] if vlims[1] is not None else None,
                    convergence=optimizer.converged
                )
    # Average MSE across theta0s
    mean_MSE_grid = MSE_grid.mean(axis=0)
    # Finding optimum
    arg_best_MSE = np.unravel_index(
        np.nanargmin(mean_MSE_grid),
        np.shape(mean_MSE_grid)
    )
    if verbose:
        if len(theta0) > 1:  # Finding std of mean
            std_MSE_grid = MSE_grid.std(axis=0) / np.sqrt(len(theta0)-1)
        else:
            MSE_stds = np.zeros_like(mean_MSE_grid)
        print(f"Best MSE value of {optimizer.method}: "
              f"{mean_MSE_grid[arg_best_MSE]:.4} "
              f"+- {std_MSE_grid[arg_best_MSE]:.2} "
              f"with lmbda {lmbdas[arg_best_MSE[0]]:.1E} "
              f"lrate {learning_rates[arg_best_MSE[1]]:.2} "
              f"({time()-start_time:.2f} s)")

    fig, ax = plt.subplots()
    # Plot heatmap first
    sns.heatmap(
        mean_MSE_grid,
        vmin=vlims[0], vmax=vlims[1],
        annot=True,
        fmt=".3",
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


def analytical_lmbda_plot(data_train: Data, data_val: Data, lmbdas: np.ndarray,
                          ylims: tuple = (0, None),
                          verbose: bool = False,
                          filename: str = None) -> None:
    """Plot the validation MSE of analytical linear regression parameters of
    the ridge cost function as function of lmbdas.

    Args:
        data_train (Data): Data to optimize the parameters on.
        data_val (Data): Data to evaluate the MSE of the optimal parameters on.
        lmbdas (np.ndarray): Array of the lmbda-values to use
        verbose (bool, optional): Whether to print the best result.
                                  Defaults to False.
        filename (str, optional): Where to save the figure.
                                     Defaults to None.
    """
    # Unpack data
    y_train, X_train = data_train.unpacked()
    y_val, X_val = data_val.unpacked()

    MSE_array = np.zeros(len(lmbdas))
    for i, lmbda in enumerate(lmbdas):
        # Cumbersome analytic expression using pseudo-inverse for safety
        theta_ana = ridge_solution(X_train, y_train, lmbda)
        MSE_array[i] = MSE(X_val@theta_ana, y_val)

    if verbose:
        arg_best_MSE = MSE_array.argmin()
        print(f"Best ridge MSE: {MSE_array[arg_best_MSE]:.4} "
              f"with lmbda {lmbdas[arg_best_MSE]:.1E}")
    # Plotting
    plt.plot(np.log10(lmbdas), MSE_array, label="Analytical ridge solution")
    # The analytic OLS solution
    theta_OLS = OLS_solution(X_train, y_train)
    MSE_OLS = MSE(X_val@theta_OLS, y_val)
    if verbose:
        print(f"Best OLS MSE: {MSE_OLS:.4}")
    # Plotting this as a horizontal, grey line
    plt.hlines(MSE_OLS,
               xmin=np.log10(lmbdas).min(),
               xmax=np.log10(lmbdas).max(),
               label="Analytical OLS solution",
               ls="--",
               colors="grey"
               )
    plt.ylim(ylims)
    plt.legend(loc="upper left")
    plt.xlabel(r"$\log_{10}(\lambda)$")
    plt.ylabel("Validation MSE")
    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def plot_by_iteration(data_train: Data, data_val: Data,
                      optimizers: tuple, optimizer_names: tuple,
                      max_iter: int, theta0: tuple,
                      random_state: int = None,
                      verbose: bool = False,
                      filename: str = None) -> None:
    """Plot the history by iteration of the validation MSE
    as parameters of OLS cost function are optimised using given optimizers.

    Args:
        data_train (Data): Data instance with training data.
        data_val (Data): Data instance with validation data.
        optimizers (tuple): tuple of GradientDescent instances.
        optimizer_names (tuple): Labels for the methods.
        max_iter (int): Number of iterations to plot.
        theta0 (tuple): nd.array or tuple of nd.arrays of
                        initial parameters descend from
        random_state (int, optional): np.random.seed to set. Defaults to None.
        verbose (bool, optional): Whether to print best MSE values.
                                  Defaults to False.
        filename (str, optional): Filename for saving plot. Defaults to None.
    """
    # Wrapping theta0 if not a tuple/list
    if not isinstance(theta0, (tuple, list)):
        theta0 = (theta0,)

    y_val, X_val = data_val.unpacked()

    MSE_array = np.zeros((len(optimizers), len(theta0), max_iter))
    all_iters = np.arange(max_iter)
    for i, optimizer in enumerate(optimizers):
        # Iterating through optimizers
        for j, x0 in enumerate(theta0):
            # Iterating through starting parameters
            # All available idcs in order
            all_data_idcs = np.arange(len(data_train))
            if random_state is not None:
                # Setting random_state to be same for every descent
                np.random.seed(random_state)
            optimizer._initialize(optimizer, x0)
            optimizer._it = 0
            for it in all_iters:
                # Doing iterations
                optimizer._it += 1
                do_GD_iteration(optimizer,
                                OLS_gradient,
                                data_train,
                                all_data_idcs)
                # Clipping exploded gradients resulting in poor MSEs
                MSE_array[i, j, it] = clip_exploded_gradients(
                    val=MSE(X_val@optimizer.x, y_val),
                    lim=8,  # This is semi-arbitrarily hardcoded
                    convergence=1  # Disregarding convergence
                )

    fig, axes = plt.subplots(
        nrows=2,
        gridspec_kw={'height_ratios': [2, 1]},
        sharex=True
    )
    # Plotting mean MSEs across starting points
    for MSEs, name, color, optimizer in zip(MSE_array, optimizer_names,
                                            plot_utils.colors, optimizers):
        lrate = optimizer.params["eta"](0)
        MSE_means = MSEs.mean(axis=0)
        axes[0].plot(all_iters, MSE_means,
                     lw=2, alpha=0.8, color=color,
                     label=name + fr" $\eta={lrate:.2}$")
        if verbose:
            arg_best_MSE = np.nanargmin(MSE_means)
            print(f"{name} best MSE: {MSE_means[-1]:.4} "
                  f"(it {arg_best_MSE+1})")
        if len(theta0) > 1:  # Adding 95% confidence interval
            MSE_stds = MSEs.std(axis=0) / np.sqrt(len(theta0)-1)
            axes[1].plot(all_iters, np.log10(MSE_stds),
                         color=color, lw=1, alpha=0.5)

    axes[0].legend(
        bbox_to_anchor=(0, 1.13, 1, 0.2),
        loc="upper left",
        ncol=2,
        mode="expand"
    )
    axes[1].set(xlabel="Iteration", ylabel=r"$\log_{10}(\mathrm{std})$")
    axes[0].set(ylabel="Validation MSE")
    # Hardcoded lims
    axes[0].set_ylim((0, 0.8))
    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


if __name__ == "__main__":
    # Import data
    from sknotlearn.datasets import make_FrankeFunction
    random_state = 321
    D = make_FrankeFunction(n=600, noise_std=0.1, random_state=random_state)
    D = D.polynomial(degree=5, with_intercept=False)
    D_train, D_val = D.train_test_split(ratio=3/4, random_state=random_state)
    D_train = D_train.scaled(scheme="Standard")
    D_val = D_train.scale(D_val)

    # Setting some general params
    np.random.seed(random_state)
    theta0 = [np.random.randn(D_val.n_features) for _ in range(5)]
    max_iter = 500
    batch_size = 200

    # All the gradient descent instances
    GD = GradientDescent("plain", dict(eta=0.),
                         its=max_iter)
    mGD = GradientDescent("momentum", dict(gamma=0.8, eta=0.13027),
                          its=max_iter)
    SGD = SGradientDescent("plain", dict(eta=0.06911),
                           epochs=max_iter, batch_size=batch_size,
                           random_state=random_state)
    mSGD = SGradientDescent("momentum", dict(gamma=0.8, eta=0.11915),
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
    aSGD = SGradientDescent("adagrad", dict(eta=0.49686),
                            epochs=max_iter, batch_size=batch_size,
                            random_state=random_state)
    maSGD = SGradientDescent("adagrad_momentum", dict(gamma=0.8, eta=0.46464),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    rmSGD = SGradientDescent("rmsprop", dict(eta=0.01944, beta=0.9),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)
    adSGD = SGradientDescent("adam", dict(eta=0.31974, beta1=0.9, beta2=0.99),
                             epochs=max_iter, batch_size=batch_size,
                             random_state=random_state)

    #####################
    # lmbda plot params #
    #####################
    params1 = {
        "PGD_MGD_PSGD_MSGD": dict(
            learning_rates=np.linspace(0.001, 0.14, 101),
            optimizers=(GD, mGD, SGD, mSGD),
            optimizer_names=(f"Plain GD", f"Momentum GD",
                             "Plain SGD", "Momentum SGD"),
            ylims=(0, 0.8)
        ),
        "SGD_batches_epochs": dict(
            learning_rates=np.linspace(0.001, 0.08, 101),
            optimizers=(SGD, SGDb, SGDe, SGDbe),
            optimizer_names=(f"SGD", fr"SGD, $4\times$batches",
                             fr"SGD, $4\times$epochs",
                             fr"SGD, $4\times$batches/epochs"),
            ylims=(0, 0.6)
        ),
        "adagrad": dict(
            learning_rates=np.linspace(0.001, 0.6, 101),
            optimizers=(aGD, maGD, aSGD, maSGD),
            optimizer_names=(f"AdaGrad GD", f"AdaGradMom GD",
                             "AdaGrad SGD", "AdaGradMom SGD"),
            ylims=(0, 0.8)
        ),
        "momentum": dict(
            learning_rates=np.linspace(0.001, 0.08, 101),
            optimizers=(mSGD2, mSGD3, mSGD4, mSGD, mSGD5),
            optimizer_names=(r"mSGD $\gamma=0.01$", r"mSGD $\gamma=0.1$",
                             r"mSGD $\gamma=0.5$", r"mSGD $\gamma=0.8$",
                             r"mSGD $\gamma=1$"),
            ylims=(0, 0.8)
        ),
        "tunable": dict(
            # funky learning rate to get more small eta evaluations
            learning_rates=(np.linspace(0., 0.6**(1/2), 101)**2)[1:],
            optimizers=(rmSGD, adSGD, aSGD, maSGD),
            optimizer_names=("RMSprop SGD", "Adam SGD",
                             "AdaGrad SGD", "AdaGradMom SGD"),
            ylims=(0, 0.6)
        ),
        "test": dict(
            learning_rates=np.linspace(0.11915, 0.11915, 1),
            optimizers=(SGradientDescent(method="momentum",
                                         params=dict(eta=0.11915, gamma=0.8),
                                         epochs=500, batch_size=200,
                                         random_state=random_state),),
            optimizer_names=("Momentum SGD",),
            ylims=(0, 0.8)
        ),
    }
    #######################
    # heatmap plot params #
    #######################
    params2 = {
        "plain_GD": dict(
            learning_rates=np.linspace(0., 0.08, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=GD,
            vlims=(None, 0.4),
        ),
        "momentum_GD": dict(
            learning_rates=np.linspace(0., 0.14, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=mGD,
            vlims=(None, 0.6),
        ),
        "plain_SGD": dict(
            learning_rates=np.linspace(0., 0.05, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=SGD,
            vlims=(None, 0.4)
        ),
        "momentum_SGD": dict(
            learning_rates=np.linspace(0., 0.13, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=mSGD,
            vlims=(None, 0.6),
        ),
        "adagrad_SGD": dict(
            learning_rates=np.linspace(0., 1., 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=aSGD,
            vlims=(None, 0.4)
        ),
        "adagrad_momentum_SGD": dict(
            learning_rates=np.linspace(0., 0.6, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=maSGD,
            vlims=(None, None),
        ),
        "adam_SGD": dict(
            learning_rates=np.linspace(0., 0.45, 11)[1:],
            lmbdas=np.logspace(-8, 1, 10),
            optimizer=adSGD,
            vlims=(None, 0.7),
        )
    }

    # Choosing plot to plot
    plot1 = ""  # dict key for params1 or empty string
    plot2 = ""  # dict key for params2 or empty string
    plot3 = 0  # True or False
    plot4 = 1  # True or False

    # Plotting
    if plot1:
        tune_learning_rate(
            data_train=D_train,
            data_val=D_val,
            theta0=theta0,
            verbose=True,
            **params1[plot1],
            filename="learning_rates_"+plot1
        )

    if plot2:
        tune_lambda_learning_rate(
            data_train=D_train,
            data_val=D_val,
            theta0=theta0,
            verbose=True,
            **params2[plot2],
            filename="lmbda_learning_rates_"+plot2
        )

    if plot3:
        analytical_lmbda_plot(
            D_train, D_val,
            lmbdas=np.logspace(-8, 0, 81),
            verbose=True,
            filename="lmbda_plot_ana"
        )

    if plot4:
        plot_by_iteration(
            D_train, D_val,
            optimizers=(mGD, SGD, mSGD, adSGD),
            optimizer_names=("Mom GD", "Plain SGD",
                             "Mom SGD", "Adam SGD"),
            max_iter=max_iter,
            theta0=theta0,
            random_state=random_state,
            verbose=True,
            filename="GD_history"
        )
