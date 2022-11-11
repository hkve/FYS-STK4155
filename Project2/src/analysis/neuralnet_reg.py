"The analysis for the application of neural networks on regression."

import autograd.numpy as np
from autograd import grad, elementwise_grad
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils
from plot_utils import make_figs_path
import time 

import context
import sknotlearn.optimize as opt
from sknotlearn.data import Data
from sknotlearn.neuralnet import NeuralNetwork
from sknotlearn.datasets import make_debugdata, make_FrankeFunction, plot_FrankeFunction, load_Terrain, plot_Terrain

def introducing_act():
    """Introducing the activation functions through plots.
    """
    x = np.linspace(-10,10,1000)
    sigmoid = 1/(1+np.exp(-x))
    relu = np.maximum(np.zeros_like(x),x)
    leaky = np.maximum(0.1*x, x)
    tanh = np.tanh(x)

    act_funcs = ["sigmoid", "relu", "leaky_relu", "tanh"]

    fig, axes = plt.subplots(2,2)
    axes[0,0].plot(x, sigmoid)
    axes[0,0].set_title("sigmoid")
    axes[0,0].tick_params(labelsize=14)
    axes[1,0].plot(x, tanh, color=plot_utils.colors[1])
    axes[1,0].set_title("tanh")
    axes[1,0].tick_params(labelsize=14)
    axes[0,1].plot(x, relu, color=plot_utils.colors[2])
    axes[0,1].set_title("relu")
    axes[0,1].tick_params(labelsize=14)
    axes[1,1].plot(x, leaky, color=plot_utils.colors[3])
    axes[1,1].set_title("leaky relu")
    axes[1,1].tick_params(labelsize=14)
    plt.savefig(make_figs_path("introducing_acts.pdf"))
    plt.show()

def activation_func_2d():
    """ Creating networks with different activation functions, wanting to see the various approximations. Training on data of x**2 with noise.   
    For discussion: relu has a point of non-linearity which can be shuffeled around by the bias. Having more nodes creates more points of non-linearities. Linear is only linear. 
    """
    x, y, X = make_debugdata(n = 600, random_state=321)
    D = Data(y,x.reshape(-1,1))
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=42)

    nodes = ((100,), 1)
    eta = 0.001
    epochs = 200
    batch_size = 2**6

    SGD = opt.SGradientDescent(
        method = "adam",
        params = {"eta":eta, "beta1":0.9, "beta2":0.99},
        epochs=epochs,
        batch_size=batch_size,
        random_state=321
    )

    plt.scatter(D_test.X[:,0][::5], D_test.y[::5], color=plot_utils.colors[-1], label='Datapoints', s=10)
    act_funcs = ["sigmoid", "relu", "leaky_relu", "tanh", "linear"]
    for i, act in enumerate(act_funcs):
        NN = NeuralNetwork(
            SGD, 
            nodes, 
            random_state=321,
            cost_func="MSE",
            # lmbda=0.001,
            activation_hidden=act,
            activation_output="linear"
            )    
    
        NN.train(D_train, trainsize=1)
        y_pred = NN.predict(D_test.X)

        mse = MSE(D_test.y, y_pred)
        print(f"{act} gives {mse = :.6f}")

        sorted_idcs = D_test.X[:,0].argsort()
        plt.plot(D_test.X[sorted_idcs,0], y_pred[sorted_idcs], color=plot_utils.colors[i], label=act)
    plt.legend()    
    plt.show()


def plot_NN_vs_test(D_train, D_test, eta, nodes, epochs=800, batch_size=80, random_state=321, filename_test=None, filename_pred=None):
    SGD = opt.SGradientDescent(
        method = "adam",
        params = {"eta":eta, "beta1":0.9, "beta2":0.99},
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state
    )

    NN = NeuralNetwork(
        SGD, 
        nodes, 
        random_state=random_state,
        cost_func="MSE",
        # lmbda=0.001,
        activation_hidden="sigmoid",
        activation_output="linear"
        )

    NN.train(D_train, trainsize=1)
    y_pred = NN.predict(D_test.X)

    print(MSE(D_test.y, y_pred))

    #Plot: 
    D_pred = Data(y_pred, D_test.X)
    D_pred = D_train.unscale(D_pred)

    D_test = D_train.unscale(D_test)

    plot_Terrain(D_test, angle=(16,-165), filename=filename_test)
    plot_Terrain(D_pred, angle=(16,-165), filename=filename_pred)


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

    MSE_grid = list()
    for eta in etas: 
        MSE_list = list()
        SGD = opt.SGradientDescent(
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
            y_pred = NN.predict(D_test.X)

            MSE_list.append(MSE(D_test.y, y_pred))
        MSE_grid.append(MSE_list)
        
    run_time = time.time() - start_time 
    print(f"{epochs = } and {batch_size = } gives {run_time = :.3f} s")

    
    # Ploting the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(MSE_grid, annot=True, cmap="viridis", ax=ax, cbar_kws={'label':'MSE'})
    # Plot a red triangle around best value. I think this code is Messi<3
    arg_best_MSE = np.unravel_index(np.nanargmin(MSE_grid), np.shape(MSE_grid))
    ax.add_patch(plt.Rectangle((arg_best_MSE[1], arg_best_MSE[0]), 1, 1, fc='none', ec='red', lw=5, clip_on=False))

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
        plt.savefig(make_figs_path(filename), dpi=300)
    # plt.show()


def MSE(y_test, y_pred):
    assert y_test.shape == y_pred.shape, f"y and y_pred have different shapes. {y_test.shape =}, {y_pred.shape =}"
    return np.mean((y_test - y_pred)**2)



if __name__ == "__main__":
    epochs = 500
    batch_size = 15

    D = load_Terrain(random_state=123, n=600)
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=42)
    D_train = D_train.scaled(scheme="Standard")    
    D_test = D_train.scale(D_test)
    y_test = D_test.y   

    # nodes = [10,20,50]
    # etas = [0.001, 0.01, 0.08]
    # layers = [1]
    # nodes = [2, 10, 20, 50, 100, 200]
    # etas = [0.0005, 0.001, 0.01, 0.06, 0.08, 0.1, 0.8]
    # layers = [1,3,5]
    # for l in layers:
    #     filename = f"nodes_etas_heatmap_{l}.pdf"
    #     nodes_etas_heatmap(
    #         D_train=D_train, 
    #         D_test=D_test, 
    #         etas=etas, 
    #         nodes=nodes, 
    #         layers=l, 
    #         epochs=epochs, 
    #         batch_size=batch_size, 
    #         filename=filename
    #     )

    # nodes = ((50,)*3, 1)
    # eta = 0.001
    # plot_NN_vs_test(
    #     D_train=D_train, 
    #     D_test=D_test, 
    #     eta=eta, nodes=nodes, 
    #     epochs=epochs, 
    #     batch_size=batch_size,
    #     # filename_test="terrain_test.pdf", 
    #     # filename_pred="terrain_predicted.pdf"
    # )

    # Check activation function:
    #Plot in same plot, include MSE 
    # activation_func_2d()

    introducing_act()
    



    "SGD"
    #Interesting parameters:
    eta_plain = [0.001, 0.01, 0.1, 0.9] #The best is eta=0.1
    gamma_momentum = 0.8 #The best when eta=0.1

    "Layers and nodes"