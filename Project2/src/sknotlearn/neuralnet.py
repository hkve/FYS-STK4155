import numpy as np
from autograd import grad

import optimize as opt
from data import Data

class NeuralNetwork:
    """_summary_"""
    def __init__(
        self, 
        optimizer:opt.GradientDescent,
        nodes:tuple[int],
        activationHidden:str="sigmoid",
        activationOutput:str="sigmoid"
        ) -> None:
        """
        Args:
            optimizer (GradientDescent):
            nodes (tuple[int]): 
        """
        self.n_hidden_nodes, self.n_output_nodes = nodes
        self.n_hidden_layers = len(self.n_hidden_nodes)

        self.weights = [None]*self.n_hidden_layers
        self.biases = [None]*self.n_hidden_layers

        self._activationHidden = self.activation_functions[activationHidden]
        self._activationOutput = self.activation_functions[activationOutput]

    def init_biases_and_weights(self) -> None:
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