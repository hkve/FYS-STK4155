import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils
import time

import context
from sknotlearn.data import Data
from sknotlearn.datasets import load_BreastCancer
from sknotlearn.optimize import SGradientDescent
from sknotlearn.neuralnet import NeuralNetwork


def network_trainer(**kwargs):
    GB_defaults = {
        "method": "plain",
        "params": {"eta":0.05}, 
        "epochs": 100,
        "batch_size": 20,
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
        if k in NN_defaults.keys():
            NN_defaults[k] = v 

    SGD = SGradientDescent(**GB_defaults)
            
    NN = NeuralNetwork(optimizer=SGD, **NN_defaults)

    return NN

def varying_activation_functions(D_train, D_test, activation_functions = ["linear", "sigmoid"]):
    etas = np.linspace(0.01, 0.7, 10)
    acc_arrays = [np.zeros_like(etas)]*len(activation_functions)

    fig, ax = plt.subplots()
    for acc_array, af in zip(acc_arrays, activation_functions):
        for i, eta in enumerate(etas):
            NN = network_trainer(params={"eta": eta}, activation_hidden=af)
            NN.train(D_train, trainsize=1)

            acc_array[i] = NN.accuracy(D_test.X, D_test.y)

        ax.plot(etas, acc_array, label=af)

    ax.set(xlabel="Learning rate", ylabel="Accuracy", ylim=(0,1))
    ax.legend()
    plt.show()

def nodes_etas_heatmap(D_train, D_test, etas, nodes, layers=2, epochs=800, batch_size=80, random_state=321, filename=None):
    """Plot the heatmap of MSE-values given various number of nodes and learning rates for a given number of layers . 

    Args:
        D_train (Data): Training data.
        D_test (Data): Testing data
        etas (ndarray, list): A list (or array) of the various learning rates. 
        nodes (ndarray, list): A list (or array) of the various number of nodes in each layer. (NB: All layers have the same number of nodes) 
        layers (int): Number of layers. Defaults to 2.
        epochs (int): Number of epochs for which each gradient is calculated. Defaults to 800.
        batch_size (int): The size of each bach in the SGD. Defaults to 80.
        random_state (int): Defaults to 321.
        filename (_type_, str): Filename of plot. Needs to be implemented to save plot. Defaults to None.
    """

    start_time = time.time()

    ACC_grid = list()
    for eta in etas: 
        ACC_list = list()
        SGD = SGradientDescent(
            method = "adam",
            params = {"eta":eta, "beta1":0.9, "beta2":0.99},
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state
        )
        for n in nodes:
            #create the nodes of the network:
            nodes_ = ((n,)*layers, 1)
            NN = NeuralNetwork(
                SGD, 
                nodes_, 
                random_state=random_state,
                cost_func="MSE",
                # lmbda=0.001,
                activation_hidden="sigmoid",
                activation_output="linear"
            )

            NN.train(D_train, trainsize=1)
            ACC_list.append(NN.accuracy(D_test.X, D_test.y))
        ACC_grid.append(ACC_list)
        
    run_time = time.time() - start_time 
    print(f"{epochs = } and {batch_size = } gives {run_time = :.3f} s")

    
    # Ploting the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(ACC_grid, annot=True, cmap="viridis", ax=ax, cbar_kws={'label':'Accuracy'})
    # Plot a red triangle around best value. I think this code is Messi<3
    arg_best_ACC = np.unravel_index(np.nanargmax(ACC_grid), np.shape(ACC_grid))
    ax.add_patch(plt.Rectangle((arg_best_ACC[1], arg_best_ACC[0]), 1, 1, fc='none', ec='red', lw=5, clip_on=False))

    # Set xylabels and ticks. This code is ronaldo, but it works.
    ax.set_xlabel("Nodes in each layer")
    ax.set_xticks(
        np.arange(len(nodes))+.5,
        labels=nodes
    )
    ax.set_ylabel("Learning rate")
    ax.set_yticks(
        np.arange(len(etas))+.5,
        labels=etas
    )
    if filename:
        plt.savefig(plot_utils.make_figs_path(filename), dpi=300)
    plt.show()

def main():
    D_train, D_test = breast_cancer_data()

    SGD = SGradientDescent(
        method = "adam",
        params = {"eta":0.05, "beta1": 0.9, "beta2": 0.99},
        epochs=100,
        batch_size=20,
        random_state=321
    )

    nodes = ((20, 20, ), 1)
    NN = NeuralNetwork(
        SGD, 
        nodes=nodes, 
        random_state=321,
        cost_func="BCE",
        activation_hidden="sigmoid",
        activation_output="sigmoid",
        lmbda=0.000001
    )


    NN.train(D_train, trainsize=1)
    print(NN.accuracy(D_test.X, D_test.y))

    y_pred, proba =  NN.classify(D_test.X, return_prob=True)

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


def breast_cancer_data(random_state=321):
    D = load_BreastCancer()
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    _, X_train = D_train.unpacked()
    _, X_test = D_test.unpacked()

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    D_train.X = X_train
    D_test.X = X_test

    return D_train, D_test


if __name__ == "__main__":
    # test_logreg()
    # main()

    D_train, D_test = breast_cancer_data()

    varying_activation_functions(D_train, D_test, activation_functions=["relu", "leaky_relu", "tanh", "sigmoid"])

    # etas = [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # nodes = [1,5,10,15,20,30,40,50]

    # nodes_etas_heatmap(D_train, D_test, etas, nodes, layers=2, epochs=100, batch_size=20)