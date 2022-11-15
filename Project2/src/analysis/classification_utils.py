import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils

import context
from sknotlearn.optimize import SGradientDescent
from sknotlearn.neuralnet import NeuralNetwork

def round_nearest(arg, decimals=2):
    labels = [""]*len(arg)
    for i, num in enumerate(arg):
        labels[i] = str(round(num, 2))

    return labels


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

        ax.plot(etas, acc, label=label, marker=".", linestyle="-", markersize=20)

    ax.set(xlabel="Learning rate", ylabel="Accuracy")
    ax.legend()
    
    if filename:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def lmbda_eta_heatmap(D_train, D_test, etas, lmbdas, nodes=((1,), 1), filename=None):
    etas = np.logspace(*etas)
    lmbdas = np.logspace(*lmbdas)
    ACC_grid = list()
    for lmbda in lmbdas: 
        ACC_list = list()
        for eta in etas:
            NN = network_trainer(nodes=nodes, params={"gamma":0.8, "eta":eta}, lmbda=lmbda, activation_hidden="leaky_relu")
            NN.train(D_train, trainsize=1)

            ACC = NN.accuracy(D_test.X, D_test.y) if NN.optimizer.converged else np.nan
            print(ACC, eta, lmbda)
            ACC_list.append(ACC)

        ACC_grid.append(ACC_list)
        
    fig, ax = plt.subplots()
    sns.heatmap(ACC_grid, annot=True, cmap=plot_utils.cmap, ax=ax, cbar_kws={'label':'Accuracy'}, vmin=0.66, vmax=1, fmt=".3")
    
    arg_best_ACC = np.unravel_index(np.nanargmax(ACC_grid), np.shape(ACC_grid))
    ax.add_patch(plt.Rectangle((arg_best_ACC[1], arg_best_ACC[0]), 1, 1, fc='none', ec='red', lw=5, clip_on=False))

    # Set xylabels and ticks. This code is ronaldo, but it works.
    ax.set_xlabel("Learning rate")
    ax.set_xticks(np.arange(len(etas))+.5, labels=np.log10(etas))
    ax.set_ylabel(r"log$_{10}(\lambda)$")
    ax.set_yticks(np.arange(len(lmbdas))+.5, labels=np.log10(lmbdas))
    
    if filename:
        plt.savefig(plot_utils.make_figs_path(filename))
    plt.show()


def main():
    D_train, D_test = breast_cancer_data()

    SGD = SGradientDescent(
        method = "adagrad_momentum",
        params = {"gamma":0.8, "eta":0.008},
        epochs=100,
        batch_size=20,
        random_state=321
    )

    nodes = ((5, ), 1)
    NN = NeuralNetwork(
        SGD, 
        nodes=nodes, 
        random_state=321,
        cost_func="BCE",
        activation_hidden="linear",
        activation_output="sigmoid",
    )

    NN.train(D_train, trainsize=1)
    print(SGD.converged, NN.optimizer.converged)
    print(NN.accuracy(D_test.X, D_test.y))

    y_pred, proba =  NN.classify(D_test.X, return_prob=True)

    print(f"w, b: {NN.weights[0]}, {NN.biases[0]}")
    for a, b, c in zip(y_pred, D_test.y, proba):
        print(a, b, c)


def test_logreg():
    from sklearn.linear_model import LogisticRegression
    D = load_BreastCancer()

    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    clf = LogisticRegression(max_iter=10000).fit(D_train.X, D_train.y)

    y_pred = clf.predict(D_test.X)
    

    acc = np.sum(y_pred == D_test.y)/len(y_pred)
    print(acc)

