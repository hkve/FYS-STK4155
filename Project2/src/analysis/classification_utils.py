import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils

import context
from sknotlearn.optimize import SGradientDescent, GradientDescent
from sknotlearn.neuralnet import NeuralNetwork
from sknotlearn.logreg import LogisticRegression as LogReg

def round_nearest(arg, decimals=2, sig=False):
    labels = [""]*len(arg)

    for i, num in enumerate(arg):
        if sig:
            s = -int(np.floor(np.log10(num)))
            labels[i] = str(round(num, s+decimals-1))
        else:
            labels[i] = str(round(num, 2))

    return labels

def shoter_labels(arg, n=5):
    ticks = np.arange(len(arg))+.5
    labels = arg
    
    if len(arg) > n:
        skip = int(len(arg)/n) 
        labels = arg[::skip]
        ticks = np.arange(len(labels)+2*skip, step=skip)+.75

    return ticks, labels

def network_trainer(**kwargs):
    GB_defaults = {
        "method": "adagrad_momentum",
        "params": {"gamma":0.8, "eta":0.1}, 
        "epochs": 100,
        "batch_size": 200,
        "random_state": 321,
    }
    NN_defaults = {
        "nodes": ((4, ), 1), 
        "cost_func": "BCE",
        "lmbda": None,
        "activation_hidden": "sigmoid",
        "activation_output": "sigmoid"
    }

    for k, v in kwargs.items():
        if k in GB_defaults.keys():
            GB_defaults[k] = v
        elif k in NN_defaults.keys():
            NN_defaults[k] = v 
        else:
            print(f"{k} not present in NN or SGB, {v}")
            exit()

    SGD = SGradientDescent(**GB_defaults)
            
    NN = NeuralNetwork(optimizer=SGD, **NN_defaults)

    return NN


def varying_activation_functions(D_train, D_test, 
                                activation_functions = ["sigmoid", "tanh"],
                                nodes = ((5,5,),1),
                                eta_range = (0.01, 0.4, 10),
                                ylim=None, 
                                filename=None
                                ):
    etas = np.linspace(*eta_range)

    cut = False
    fig, ax = plt.subplots()
    for i, af in enumerate(activation_functions):
        acc = np.zeros_like(etas)

        for j, eta in enumerate(etas):
            NN = network_trainer(params={"eta":eta, "gamma":0.8}, activation_hidden=af, nodes=nodes)
            NN.train(D_train, trainsize=1)

            if NN.optimizer.converged:
                acc[j] = NN.accuracy(D_test.X, D_test.y)
            else:
                acc[j] = np.nan

        label = af.replace("_", " ").capitalize()

        ax.plot(etas, acc, label=label, marker=plot_utils.markers[i], linestyle="-", markersize=10, alpha=0.8)

    ax.set(xlabel="Learning rate", ylabel="Accuracy")
    ax.legend(ncol=2, loc="lower center")
    
    if ylim: ax.set(ylim=ylim)
    if filename: plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def lmbda_eta_heatmap(D_train, D_test, etas, lmbdas, nodes=((1,), 1), activation_hidden="sigmoid", filename=None):
    etas = np.linspace(*etas)
    lmbdas = np.logspace(*lmbdas)
    ACC_grid = list()
    for lmbda in lmbdas: 
        ACC_list = list()
        for eta in etas:
            NN = network_trainer(nodes=nodes, params={"gamma":0.8, "eta":eta}, lmbda=lmbda, activation_hidden=activation_hidden)
            NN.train(D_train, trainsize=1)

            ACC = NN.accuracy(D_test.X, D_test.y) if NN.optimizer.converged else np.nan
            ACC_list.append(ACC)

        ACC_grid.append(ACC_list)
        
    ACC_grid = np.array(ACC_grid)

    fig, ax = plt.subplots()
    sns.heatmap(ACC_grid, annot=True, cmap=plot_utils.cmap, ax=ax, cbar_kws={'label':'Accuracy'}, fmt=".3")
    
    arg_best_ACC = np.unravel_index(np.nanargmax(ACC_grid), np.shape(ACC_grid))
    ib, jb = arg_best_ACC
    ij_bests = np.argwhere(ACC_grid==ACC_grid[ib,jb])

    for ij in ij_bests:
        i, j = ij
        print(f"Best ACC = {ACC_grid[i][j]}, lmbda = {lmbdas[i]:.4e}, eta = {etas[j]:.4e}")
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fc='none', ec='red', lw=5, clip_on=False))

    # Set xylabels and ticks. This code is ronaldo, but it works.
    ax.set_xlabel("Learning rate")
    xticks, xlabels = shoter_labels(round_nearest(etas, decimals=1, sig=True))
    ax.set_xticks(xticks, labels=xlabels)


    ax.set_ylabel(r"log$_{10}(\lambda)$")
    yticks, ylabels = shoter_labels(np.log10(lmbdas))
    ax.set_yticks(yticks, labels=ylabels)
    
    if filename: plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def logreg_different_activations(D_train, D_test, eta_range, opts, labels, filename=None):
    etas = np.linspace(*eta_range)

    fig, ax = plt.subplots()
    for i, opt in enumerate(opts):
        ACC = np.zeros_like(etas)
        for j, eta in enumerate(etas):
            opt.params["eta"] = lambda it: eta
            clf = LogReg().fit(D_train, opt)
            ACC[j] = clf.accuracy(D_test.X, D_test.y)
            print(clf.converged)
        ax.plot(etas, ACC, label=labels[i])

    ax.set(xlabel="Learning rate", ylabel="Accuracy")
    ax.legend()
    if filename: plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def logreg_with_sklearn(D_train, D_test):
    from sklearn.linear_model import LogisticRegression

    GD = GradientDescent(method= "plain", params= {"eta":0.1},  its=1000)

    clf1 = LogReg().fit(D_train, optimizer=GD)
    clf2 = LogisticRegression(max_iter=10000).fit(D_train.X, D_train.y)

    acc1 = clf1.accuracy(D_test.X, D_test.y)
    acc2 = np.sum(clf2.predict(D_test.X) == D_test.y)/D_test.n_points

    print(f"sknotlearn {acc1 = }, sklearn {acc2 = }")