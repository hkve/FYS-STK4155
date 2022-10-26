import numpy as np
from autograd import grad

import optimize as opt
from data import Data

class NeuralNetwork:
    """_summary_"""
    def __init__(
        self, 
        optimizer:opt.GradientDescent,
        nodes:tuple,
        activation_hidden:str="sigmoid",
        activation_output:str="sigmoid"
        ) -> None:
        """
        Args:
            optimizer (GradientDescent):
            nodes (tuple[int]): 
            activation_hidden (str):
            activation_output (str):
        """
        self.n_hidden_nodes, self.n_output_nodes = nodes
        self.n_hidden_layers = len(self.n_hidden_nodes)

        self.weights = [None]*self.n_hidden_layers
        self.biases = [None]*self.n_hidden_layers

        self._activation_hidden = self.activation_functions[activation_hidden]
        self._activation_output = self.activation_functions[activation_output]

    def _init_biases_and_weights(self) -> None:
        pass

    def _backprop_pass(self) -> None:
        pass

    def _forward_pass(self) -> None:
        pass

    def _sigmoid(x):
        return 1/(1+np.exp(-x))

    def predict(self, X:np.array) -> np.array:
        y_pred = None

        return y_pred

    def train(self, D:Data, trainsize=2/3):
        self.D_train, self.D_test = D.train_test_split(ratio=trainsize)

        # Call forward and backprop ...
    
    activation_functions = {
        "sigmoid": _sigmoid, # Add ReLU and such...    
    }

if __name__ == "__main__":
    GD = opt.GradientDescent(
        method = "plain",
        params = {"eta":0.8},
        its=100
    )

    nodes = ((10, 10, 10), 1)
    NN = NeuralNetwork(GD, nodes)

    print(NN._activationOutput(0))