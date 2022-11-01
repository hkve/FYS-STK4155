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

        self.optimizer = optimizer

        self.n_hidden_nodes, self.n_output_nodes = nodes
        self.n_hidden_layers = len(self.n_hidden_nodes)

        self.weights = [None]*(self.n_hidden_layers + 1)
        self.biases = [None]*(self.n_hidden_layers + 1)
        self.zs = [None]*(self.n_hidden_layers + 1)

        self.cost_func = self.cost_funcitons[cost_func]

        self._activation_hidden = self.activation_functions[activation_hidden]
        self._activation_output = self.activation_functions[activation_output]

        self._grad_activation_output = elementwise_grad(self._activation_output)
        self._grad_activation_hidden = elementwise_grad(self._activation_hidden)

    def _init_biases_and_weights(self) -> None:
        """NB: Weights might be initialized opposite to the convension
        """
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

    def _forward_pass(self, x:np.array) -> tuple:
        """
        
        """
        # Store z and f(z) values
        z_fwp = [None]*(self.n_hidden_layers+1)
        a_fwp = [None]*(self.n_hidden_layers+2)
        
        a = a_fwp[0] = x

        #hidden layer:
        for h in range(self.n_hidden_layers):
            z = a @ self.weights[h].T + self.biases[h]
            a = self._activation_hidden(z)

            z_fwp[h] = z
            a_fwp[h+1] = a

        #output layer: 
        z = z_fwp[-1] = a @ self.weights[-1].T + self.biases[-1]
        y = a_fwp[-1] = self._activation_output(z)
        
        return (y, z_fwp, a_fwp)


    def _backprop_pass(self, y, z_fwp, a_fwp) -> tuple:
        """
        Function to perform backwards propagation. Takes target values y, in addition
        to z and a values calculated during the forward pass. Calculates the gradient of 
        weights and biases.

        Args:
            y (np.array): Target value to use for cost function derivative evaluation
            z_fwp (List[np.array]): The output from each layer, before the activation functions is used.
            a_fwp (List[np.array]): The output from each layer, after the activation function is used. Also includes the X data
        
        Returns:
            grad_Ws (List[np.array]): List of arrays representing the weights for each layer
            grad_bs (List[np.array]): List of arrays representing the bias for each layer
        """
        
        delta_ls = [None]*(self.n_hidden_layers+1)
        grad_Ws = [None]*(self.n_hidden_layers+1) 
        grad_bs = [None]*(self.n_hidden_layers+1)

        y_pred = a_fwp[-1]

        grad_cost = elementwise_grad(lambda y_pred : self.cost_func(y, y_pred))
        delta_ls[-1] = self._grad_activation_output(z_fwp[-1])*grad_cost(y_pred)

        for i, (w,b) in enumerate(zip(self.weights, self.biases)):
            print(f"Layer({i}), W = {w.shape}, b = {b.shape}")


        for i in reversed(range(self.n_hidden_layers)):
            fp = self._grad_activation_hidden(z_fwp[i])
            W = self.weights[i+1]
            delta_prev = delta_ls[i+1]
            delta_ls[i] = delta_prev @ W * fp


        for i in reversed(range(0, self.n_hidden_layers+1)):
            grad_Ws[i] = delta_ls[i].T @ a_fwp[i]
            grad_bs[i] = delta_ls[i].sum(axis=0)


        for i, (wp,bp) in enumerate(zip(grad_Ws, grad_bs)):
            print(f"Layer({i}), W' = {wp.shape}, b' = {bp.shape}")

        return grad_Ws, grad_bs


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


        # Call forward for every datapoint: 
        n = len(self.D_train)
        y, X = self.D_train.unpacked()

        y_pred, z_fwp, a_fwp = self._forward_pass(X)
        grad_Ws, grad_bs = self._backprop_pass(y, z_fwp, a_fwp)



    #Activation funtions:
    def _no_activation(x):
        return x

    def _sigmoid(x):
        return 1/(1+np.exp(-x))
    
    def _relu(x):
        return np.maximum(np.zeros_like(x),x)

    def _leaky_relu(x):
        return np.maximum(0.1*x, x)
    
    def _tanh(x):
        return np.tanh(x)

    def _linear(x):
        return x

    #Cost functions: 
    def _MSE(y, y_pred):
        return np.mean((y - y_pred)**2)

    #Dicts:        
    activation_functions = {
        "sigmoid": _sigmoid,
        "relu": _relu,
        "leaky_relu": _leaky_relu,
        "tanh": _tanh,
        "linear": _linear, 
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

    
    nodes = ((10, 9, 8), 1)
    NN = NeuralNetwork(
        GD, 
        nodes, 
        random_state=321,
        cost_func="MSE",
        activation_output="linear"
    )
    D_train, D_test = D.train_test_split(random_state=42)
    D_train = D_train.scaled(scheme="Standard")    
    D_test = D_train.scale(D_test)
    
    NN.train(D_train)




