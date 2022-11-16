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

from sklearn.neural_network import MLPRegressor

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
    epochs = 500
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
    # plt.savefig(make_figs_path("c_activations_2d_data.pdf"))
    plt.show()

def plot_NN_vs_test(D_train, D_test, eta, nodes, batch_size, epochs=500, lmbda=None, random_state=321, filename_test=None, filename_pred=None):
    """Function for training an NN and plotting the true terrain data and the NN predicted terrain data for qualitative comparison. Will also print the MSE. The optimal parameters are likely already found in another process. 

    Args:
        D_train (Data): Training data
        D_test (Data): Testing data
        eta (float): Learning rate 
        nodes (tuple): The nodes and layers of the network. E.g. ((10,10,10),1)
        epochs (int): Number of epochs for which the gradient is calculated. Defaults to 800.
        batch_size (int): The size of each bach in the SGD. Defaults to 80.
        random_state (int): Defaults to 321.
        filename_test (_type_, str): Filename of true data plot. Needs to be implemented to save plot. Defaults to None.
        filename_pred (_type_, str): Filename of NN predicted plot. Needs to be implemented to save plot. Defaults to None.
    """
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
        lmbda=lmbda,
        activation_hidden="sigmoid",
        activation_output="linear"
        )

    NN.train(D_train, trainsize=1)
    y_pred = NN.predict(D_test.X)

    print(f"MSE = {MSE(D_test.y, y_pred) :.5f}")

    #Plot: 
    D_pred = Data(y_pred, D_test.X)
    D_pred = D_train.unscale(D_pred)

    D_test = D_train.unscale(D_test)

    plot_Terrain(D_test, angle=(16,-165), filename=filename_test)
    plot_Terrain(D_pred, angle=(16,-165), filename=filename_pred)

def nodes_etas_heatmap(D_train, D_test, etas, nodes, layers, batch_size, activation_hidden="sigmoid", epochs=500, random_state=321, filename=None):
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
                activation_hidden=activation_hidden,
                activation_output="linear"
            )

            NN.train(D_train, trainsize=1)
            y_pred = NN.predict(D_test.X)

            MSE_list.append(MSE(D_test.y, y_pred))
        MSE_grid.append(MSE_list)
        
    run_time = time.time() - start_time 
    print(f"{epochs = } and {batch_size = } gives {run_time = :.3f} s")

    v_min = np.min(MSE_grid)
    v_max = np.min(MSE_grid) * 4
    
    # Ploting the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(MSE_grid, annot=True, cmap=plot_utils.cmap, ax=ax, cbar_kws={'label':'MSE'}, vmin=v_min, vmax=v_max)
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
    else:
        plt.show()

def implement_scikit(D_train, D_test, nodes, layers, etas, epochs=500, batch_size="auto", early_stopping=False, activation="logistic", random_state=321, filename=None):
    """Creating a neural network using scikit-learn and plotting the heatmap of MSE-values given various number of nodes and learning rates for a given number of layers. 

    Args:
        D_train (Data): Training data.
        D_test (Data): Testing data
        etas (ndarray, list): A list (or array) of the various learning rates. 
        nodes (ndarray, list): A list (or array) of the various number of nodes in each layer. (NB: All layers have the same number of nodes) 
        layers (int): Number of layers. 
        epochs (int): Number of epochs for which each gradient is calculated. Defaults to 500.
        batch_size (int): The size of each bach in the SGD. Defaults to 80.
        early_stopping (bool, optional): Terminate training when validation score is not improving. Defaults to False.
        random_state (int): Defaults to 321.
        filename (_type_, str): Filename of plot. Needs to be implemented to save plot. Defaults to None.
    """


    MSE_grid = list()
    for eta in etas: 
        MSE_list = list()
        for n in nodes:
            np.random.seed(random_state)
            nodes_ = ((n,)*layers)
            dnn = MLPRegressor(
                hidden_layer_sizes=nodes_,
                activation=activation,
                solver="adam",
                learning_rate_init=eta,
                max_iter=epochs,
                batch_size=batch_size,
                early_stopping=early_stopping,
                alpha=0,
            )

            dnn.fit(D_train.X, D_train.y)
            y_pred = dnn.predict(D_test.X)

            MSE_list.append(MSE(D_test.y, y_pred))
        MSE_grid.append(MSE_list)

    v_min = np.min(MSE_grid)
    v_max = np.min(MSE_grid) * 4

    
    # Ploting the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(MSE_grid, annot=True, cmap=plot_utils.cmap, ax=ax, cbar_kws={'label':'MSE'}, vmin=v_min, vmax=v_max)
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
    else:
        plt.show()

def regularization(D_train, D_test, nodes, layers, eta, lmbdas, batch_size, epochs=500, random_state=321, filename=None):
    """Plot the MSE as a function of regularization parameter
    """
    nodes_ = ((nodes,)*layers, 1)

    SGD = opt.SGradientDescent(
        method = "adam",
        params = {"eta":eta, "beta1":0.9, "beta2":0.99},
        epochs=epochs,
        batch_size=batch_size,
        random_state=random_state
    )

    if not 0 in lmbdas:
        lmbdas_ = np.append(None, lmbdas)
    
    MSE_list = []
    for i, lmbda in enumerate(lmbdas_):
        #To keep track in the terminal: 
        print(fr"lambda nr. {i+1}/{len(lmbdas_)}")

        NN = NeuralNetwork(
            SGD, 
            nodes_, 
            random_state=random_state,
            cost_func="MSE",
            lmbda=lmbda,
            activation_hidden="sigmoid",
            activation_output="linear"
            )
        NN.train(D_train, trainsize=1)
        y_pred = NN.predict(D_test.X)

        if lmbda is None:
            OLS = MSE(D_test.y, y_pred)  
        else: 
            MSE_list.append(MSE(D_test.y, y_pred))

    #Remember to print best MSE and its lambda 
    print(fr"Best MSE: {np.nanmin(MSE_list)}" + "\n" + fr"corresponding lambda: {lmbdas[np.nanargmin(MSE_list)]}")
    
    OLS = OLS * np.ones_like(MSE_list)
    
    fig, ax = plt.subplots()
    ax.plot(np.log10(lmbdas), OLS, "--", color="grey", label="OLS solution")
    ax.plot(np.log10(lmbdas), MSE_list, label="ridge solution")
    ax.set_xlabel(r"$\log_{10} \lambda $")
    ax.set_ylabel(r"Validation MSE")
    ax.set_ylim(0)
    ax.legend()
    if filename: 
        plt.savefig(make_figs_path(filename))
    else:
        plt.show()

#Make a heatplot of eta vs lambda with R2

def MSE(y_test, y_pred):
    assert y_test.shape == y_pred.shape, f"y and y_pred have different shapes. {y_test.shape =}, {y_pred.shape =}"
    return np.mean((y_test - y_pred)**2)

def R2(y_test, y_pred):
    pass


if __name__ == "__main__":
    epochs = 500
    batch_size = 200

    D = load_Terrain(random_state=123, n=600)
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=42)
    D_train = D_train.scaled(scheme="Standard")    
    D_test = D_train.scale(D_test)
    y_test = D_test.y 


    #Heatmap:
    # nodes = [2, 10, 20, 50, 100, 200]
    # etas = [0.0005, 0.001, 0.01, 0.06, 0.08, 0.1, 0.8]
    # # layers = [1,3,5]
    # for l in layers:
    #     filename = f"nodes_etas_heatmap_{l}_lrelu.pdf"
    #     nodes_etas_heatmap(
    #         D_train=D_train, 
    #         D_test=D_test, 
    #         etas=etas, 
    #         nodes=nodes, 
    #         layers=l, 
    #         epochs=epochs, 
    #         batch_size=batch_size, 
    #         activation_hidden="leaky_relu", 
    #         filename=filename
    #     )

    #Scikit-learn
    # nodes = [2, 10, 20, 50, 100, 200]
    # etas = [0.0005, 0.001, 0.01, 0.06, 0.08, 0.1, 0.8]
    # layers = [1,3,5]
    # for l in layers:
    #     filename = f"nodes_etas_heatmap_sk_{l}.pdf"
    #     implement_scikit(
    #         D_train, 
    #         D_test, 
    #         nodes=nodes, 
    #         layers=l, 
    #         etas=etas, 
    #         batch_size=batch_size,
    #         activation="logistic",
    #         filename=filename,  
    #     )
    #Believe there are too few datapoints to implement early stopping (not enough validation data to decide when to actually stop)

    #Regularization 
    # lmbdas = np.logspace(-9, -1, 50)
    # nodes = 200
    # eta = 0.001
    # layers = 5
    # regularization(
    #     D_train, 
    #     D_test, 
    #     nodes, 
    #     layers, 
    #     eta, 
    #     lmbdas, 
    #     batch_size, 
    #     epochs=500, 
    #     random_state=321, 
    #     filename="lmbdas_NN_reg.pdf"
    # ) 


    #Plot terrain data:
    # nodes = ((200,)*5, 1)
    # lmbda = 7.906043210907685e-05
    # eta = 0.001
    # plot_NN_vs_test(
    #     D_train=D_train, 
    #     D_test=D_test, 
    #     eta=eta, 
    #     lmbda=lmbda,
    #     nodes=nodes, 
    #     epochs=epochs, 
    #     batch_size=batch_size,
    #     filename_test="terrain_test.pdf", 
    #     filename_pred="terrain_predicted.pdf"
    # )



    # Check activation functions:
    # activation_func_2d()
    # introducing_act()