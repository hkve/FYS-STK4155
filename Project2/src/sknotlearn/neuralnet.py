import autograd.numpy as np
from autograd import grad, elementwise_grad

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
        activation_output:str="linear",
        random_state = None,
        lmbda = None
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

        self.cost_func = self.cost_function[cost_func]

        self._activation_hidden = self.activation_functions[activation_hidden]
        self._activation_output = self.activation_functions[activation_output]

        self._grad_activation_output = elementwise_grad(self._activation_output)
        self._grad_activation_hidden = elementwise_grad(self._activation_hidden)

        self.lmbda = lmbda 

    def _init_biases_and_weights(self) -> None:
        """NB: Weights might be initialized opposite to the convension
        """
        self.weights[0] = np.random.randn(self.n_features,self.n_hidden_nodes[0])
        self.weights[1:-1] = [np.random.randn(self.n_hidden_nodes[layer],self.n_hidden_nodes[layer+1]) for layer in range(self.n_hidden_layers-1)]
        self.weights[-1] = np.random.randn(self.n_hidden_nodes[-1],1)
        
        self.biases[:-1] = [0.1*np.ones(nodes) for nodes in self.n_hidden_nodes]
        self.biases[-1] = np.array([0.1]) 

    def _flat_parameters(self, weights, biases):
        """The language of GD<3

        Returns:
            _type_: _description_
        """
        flat_parameters = np.zeros(self.n_parameters)
        idx = 0
        for w, b in zip(weights, biases):
            w_size = w.size
            b_size = b.size
            flat_parameters[idx:idx+w_size] = w.ravel()
            idx += w_size
            flat_parameters[idx:idx+b_size] = b.ravel()
            idx += b_size

        return flat_parameters
        
    def _curvy_parameters(self, flat_parameters): #;)
        idx = 0
        weights = [None]*len(self.weights)
        biases = [None]*len(self.biases) 

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w_size = w.size
            b_size = b.size
            weights[i] = flat_parameters[idx:idx+w_size].reshape(w.shape)
            idx += w_size
            biases[i] = flat_parameters[idx:idx+b_size].reshape(b.shape)
            idx += b_size

        return weights, biases

    def _forward_pass(self, X:np.array) -> tuple:
        """
        
        """
        # Store z and f(z) values
        z_fwp = [None]*(self.n_hidden_layers+1)
        a_fwp = [None]*(self.n_hidden_layers+2)
        
        # input layer
        a = a_fwp[0] = X

        # hidden layer:
        for h in range(self.n_hidden_layers):
            z = z_fwp[h] = a @ self.weights[h] + self.biases[h]
            a = a_fwp[h+1] = self._activation_hidden(z)

        # output layer: 
        z = z_fwp[-1] = a @ self.weights[-1] + self.biases[-1]
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

        # Error in each layer, weight and bias gradients 
        delta_ls = [None]*(self.n_hidden_layers+1)
        grad_Ws = [None]*(self.n_hidden_layers+1) 
        grad_bs = [None]*(self.n_hidden_layers+1)

        # What FWP predicted
        y_pred = a_fwp[-1]

        # Calculate the gradient of cost function corresponding to this set of y values    
        grad_cost = elementwise_grad(lambda y_pred : self.cost_func(y, y_pred)) 

        # Delta in output layer
        delta_ls[-1] = self._grad_activation_output(z_fwp[-1]) * grad_cost(y_pred)

        # Iterate backwards over hidden layers to calculate layer errors
        for i in reversed(range(self.n_hidden_layers)):
            fprime = self._grad_activation_hidden(z_fwp[i])
            W = self.weights[i+1]
            delta_prev = delta_ls[i+1]
            delta_ls[i] = delta_prev @ W.T * fprime

        # Iterate backwards over hidden layers to calculate gradients
        for i in range(self.n_hidden_layers+1):
            grad_Ws[i] = a_fwp[i].T @ delta_ls[i]
            if self.lmbda: 
                grad_Ws[i] += self.lmbda * self.weights
            grad_bs[i] = delta_ls[i].sum(axis=0)


        return grad_Ws, grad_bs


    def predict(self, X:np.array) -> np.array:
        a = X
        #hidden layer:
        for h in range(self.n_hidden_layers):
            z = a @ self.weights[h] + self.biases[h]
            a = self._activation_hidden(z)

        #output layer: 
        z = a @ self.weights[-1] + self.biases[-1]
        y_pred = self._activation_output(z)
        
        return y_pred[:,0]


    def grad(self, coef: np.array, data:Data, idcs:np.ndarray=None) -> np.array:
        data = data[idcs]
        # Reshape coef array to fit FWP/BWP
        weights, biases = self._curvy_parameters(coef)
        
        # Set them as the networks weights and biases
        self.weights = weights
        self.biases = biases


        # Perform forward and backward pass
        y_pred, z_fwp, a_fwp = self._forward_pass(data.X)
        grad_Ws, grad_bs = self._backprop_pass(data.y.reshape(-1,1), z_fwp, a_fwp)

        # Flatten gradients
        grad_coef = self._flat_parameters(grad_Ws, grad_bs)

        return grad_coef

    def train(self, D:Data, trainsize=3/4):
        self.D_train, self.D_test = D.train_test_split(ratio=trainsize,random_state=self.random_state, shuffle=False)
        self.n_features = D.n_features

        #Initialize weights and biases: 
        self._init_biases_and_weights()
        self.n_parameters = sum([weights.size + biases.size for weights, biases in zip(self.weights, self.biases)])

        # Call forward for every datapoint: 
        n = len(self.D_train)

        x0 = self._flat_parameters(self.weights, self.biases)

        # TODO: Evaluate error after FWP based on Test data for each iteration.
        coef_opt = self.optimizer.call(
            grad=self.grad, 
            x0=x0,
            args=(self.D_train,),
            all_idcs=np.arange(n)
        )

        weights, biases = self._curvy_parameters(coef_opt)
        self.weights = weights
        self.biases = biases

    #Activation funtions:
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
        assert y.shape == y_pred.shape, f"y and y_pred have different shapes. {y.shape =}, {y_pred.shape =}"
        return np.mean((y - y_pred)**2)

    #Dicts:        
    activation_functions = {
        "sigmoid": _sigmoid,
        "relu": _relu,
        "leaky_relu": _leaky_relu,
        "tanh": _tanh,
        "linear": _linear, 
    }

    cost_function = {
        "MSE": _MSE,
    }


if __name__ == "__main__":
    from datasets import make_debugdata, make_FrankeFunction, plot_FrankeFunction, load_Terrain, plot_Terrain
    def main():
        D = load_Terrain(random_state=321)
        # D = Data(y,x.reshape(-1,1))

        SGD = opt.SGradientDescent(
            method = "adam",
            params = {"eta":0.06, "beta1":0.9, "beta2":0.99},
            epochs=800,
            batch_size=50,
            random_state=321
        )

        nodes = ((40,), 1)
        NN = NeuralNetwork(
            SGD, 
            nodes, 
            random_state=321,
            cost_func="MSE",
            # lmbda=0.001,
            activation_hidden="sigmoid",
            activation_output="linear"
        )

        D_train, D_test = D.train_test_split(ratio=3/4, random_state=42)
        D_train = D_train.scaled(scheme="Standard")    
        D_test = D_train.scale(D_test)
        
        NN.train(D_train, trainsize=1)

        y_pred = NN.predict(D_test.X)
        # print(np.column_stack((y_pred,D.y)))
        print(np.mean((y_pred - D_test.y)**2))

        D_pred = Data(y_pred, D_test.X)
        D_pred = D_train.unscale(D_pred)

        plot_Terrain(D_train.unscale(D_test), angle=(16,-165))
        plot_Terrain(D_pred, angle=(16,-165))

        #One dimensional:
        # import matplotlib.pyplot as plt
        # sorted_idcs = D_test.X[:,0].argsort()
        # plt.scatter(D_test.X[:,0], D_test.y, c="r")
        # plt.plot(D_test.X[sorted_idcs,0], y_pred[sorted_idcs])
        # plt.show()
    
    main()