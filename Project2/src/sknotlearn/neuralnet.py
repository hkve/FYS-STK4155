import autograd.numpy as np
from autograd import grad, elementwise_grad
from datasets import make_debugdata

import optimize as opt
from data import Data

class NeuralNetwork:
    """_summary_"""
    def __init__(
        self, 
        optimizer:opt.GradientDescent,
        nodes:tuple,
        cost_func, 
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

        self.cost_func = self.cost_funcitons[cost_func]

        self._activation_hidden = self.activation_functions[activation_hidden]
        self._activation_output = self.activation_functions[activation_output]

    def _init_biases_and_weights(self) -> None:
        self.weights[0] = np.random.randn(self.n_hidden_nodes[0], self.n_features)
        self.weights[1:-1] = [np.random.randn(self.n_hidden_nodes[layer+1],self.n_hidden_nodes[layer]) for layer in range(self.n_hidden_layers-1)]
        self.weights[-1] = np.random.randn(1, self.n_hidden_nodes[-1])
        
        self.biases[:-1] = [0.1*np.ones(nodes) for nodes in self.n_hidden_nodes]
        self.biases[-1] = np.array([0.1]) 

    def _flat_parameters(self):
        """The language of GD<3

        Returns:
            _type_: _description_
        """
        flat_parameters = np.zeros(self.n_parameters)
        idx = 0
        for weights, biases in zip(self.weights, self.biases):
            w_size = weights.size
            b_size = biases.size
            flat_parameters[idx:idx+w_size] = weights.ravel()
            idx += w_size
            flat_parameters[idx:idx+b_size] = biases.ravel()
            idx += b_size
        return flat_parameters
        
    def _curvy_parameters(self, flat_parameters): #;)
        idx = 0
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            w_size = weights.size
            b_size = biases.size
            self.weights[i] = flat_parameters[idx:idx+w_size].reshape(weights.shape)
            idx += w_size
            self.biases[i] = flat_parameters[idx:idx+b_size].reshape(biases.shape)
            idx += b_size



    def _backprop_pass(self, y, y_pred) -> None:
        # TODO: Save deltas and update weights after the loop.
        # TODO: Look at GD 
        delta_output = np.dot(elementwise_grad(self._activation_output)(y_pred), elementwise_grad(lambda ypred : self.cost_func(y, ypred))(y_pred)) / len(y)

        grad_cost = elementwise_grad(lambda ypred : self.cost_func(y, ypred))
        grad_act_out = elementwise_grad(self._activation_output)
        deltas = np.zeros(self.n_hidden_layers + 1)
        deltas[0] = grad_cost * grad_act_out

        for h in reversed(range(self.n_hidden_layers)):
            grad_act_hid = elementwise_grad(self._activation_hidden)
            deltas[h] = self.weights[h+1].T @ deltas[h+1] * grad_act_hid

        #update weights 
        self.weights = self.weights * deltas 
              
        
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

    def predict(self, X:np.array) -> np.array:
        y_pred = None

        return y_pred

    def train(self, D:Data, trainsize=3/4):
        self.D_train, self.D_test = D.train_test_split(ratio=trainsize,random_state=self.random_state)
        self.n_features = D.n_features

        #Initialize weights and biases: 
        self._init_biases_and_weights()
        self.n_parameters = sum([weights.size + biases.size for weights, biases in zip(self.weights, self.biases)])
        flat_parameters = self._flat_parameters()
        # print(self.weights, f'\n', self.biases)
        # self._curvy_parameters(flat_parameters)
        # print(self.weights, f'\n', self.biases)

        #NB! Remember a loop here!

        # Call forward for every datapoint: 
        n = len(self.D_train)
        y, X = self.D_train.unpacked()
        y_pred = np.zeros(n)
        for i, xi in enumerate(X):
            y_pred[i] = self._forward_pass(xi)

        # and backprop ...
        # self._backprop_pass(y, y_pred)

    #Activation funtions:
    def _sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def _relu(x):
        return np.maximum(np.zeros_like(x),x)

    def _leaky_relu(x):
        if  x < 0:
            return 0.1 * x
        else: 
            return x
    
    def _tanh(x):
        return np.tanh(x)

    #Cost functions: 
    def _MSE(y, y_pred):
        return np.mean((y - y_pred)**2)

    #Dicts:        
    activation_functions = {
        "sigmoid": _sigmoid,
        "relu": _relu,
        "leaky_relu": _leaky_relu,
        "tanh": _tanh 
    }

    cost_funcitons = {
        "MSE": _MSE,
    }



if __name__ == "__main__":
    x, y, X = make_debugdata()
    D = Data(y,X)

    GD = opt.GradientDescent(
        method = "plain",
        params = {"eta":0.8},
        its=100
    )

    
    nodes = ((10, 9, 10), 1)
    NN = NeuralNetwork(
        GD, 
        nodes, 
        random_state=321,
        cost_func="MSE"
    )
    NN.train(D)


