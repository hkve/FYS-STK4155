import numpy as np
from autograd import grad
from datasets import make_debugdata

import optimize as opt
from data import Data

class NeuralNetwork:
    """_summary_"""
    def __init__(
        self, 
        optimizer:opt.GradientDescent,
        nodes:tuple,
        activation_hidden:str="sigmoid",
        activation_output:str="sigmoid", 
        random_state = None
        ) -> None:
        """
        Args:
            optimizer (GradientDescent):
            nodes (tuple[int]): 
            activation_hidden (str):
            activation_output (str):
        """
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

        self.n_hidden_nodes, self.n_output_nodes = nodes
        self.n_hidden_layers = len(self.n_hidden_nodes)

        self.weights = [None]*(self.n_hidden_layers + 1)
        self.biases = [None]*(self.n_hidden_layers + 1)

        self._activation_hidden = self.activation_functions[activation_hidden]
        self._activation_output = self.activation_functions[activation_output]

    def _init_biases_and_weights(self) -> None:
        if self.random_state:
            np.random.seed(self.random_state)

        self.weights[0] = np.random.randn(self.n_hidden_nodes[0], self.n_features)
        self.weights[1:-1] = [np.random.randn(self.n_hidden_nodes[layer+1],self.n_hidden_nodes[layer]) for layer in range(self.n_hidden_layers-1)]
        self.weights[-1] = np.random.randn(1, self.n_hidden_nodes[-1])

        self.biases[:-1] = [0.1*np.ones(nodes) for nodes in self.n_hidden_nodes]
        self.biases[-1] = np.array([0.1]) 

    def _backprop_pass(self) -> None:
        pass

    def _forward_pass(self, x) -> None:
        #hidden layer: 
        a = x
        for h in range(self.n_hidden_layers):
            z = self.weights[h] @ a + self.biases[h]
            a = self._activation_hidden(z)

        #output layer: 
        z = self.weights[-1] @ a + self.biases[-1]
        y = self._activation_output(z)
        return y

    def _sigmoid(x):
        return 1/(1+np.exp(-x))

    def predict(self, X:np.array) -> np.array:
        y_pred = None

        return y_pred

    def train(self, D:Data, trainsize=3/4):
        self.D_train, self.D_test = D.train_test_split(ratio=trainsize,random_state=self.random_state)
        self.n_features = D.n_features

        #Initialize weights and biases: 
        self._init_biases_and_weights()

        # Call forward for every datapoint: 
        n = len(self.D_train)
        y = np.zeros(n)
        for i, xi in enumerate(self.D_train.X):
            y[i] = self._forward_pass(xi)

        # and backprop ...
    
    activation_functions = {
        "sigmoid": _sigmoid, # Add ReLU and such...    
    }

if __name__ == "__main__":
    x, y, X = make_debugdata()
    D = Data(y,X)

    GD = opt.GradientDescent(
        method = "plain",
        params = {"eta":0.8},
        its=100
    )

    nodes = ((10, 10, 10), 1)
    NN = NeuralNetwork(GD, nodes, random_state=321)
    NN.train(D)
