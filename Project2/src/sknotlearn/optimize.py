import numpy as np
from sys import float_info, exit
EPSILON = float_info.epsilon**0.5
class GradientDescent:
    """Implements Gradient Descent minimization of a problem defined by the gradient g of scalar function wrt. argument(s) x. Implemented update rules are:
    "plain" (eta): Ordinary GD with a learning rate eta.
    "momentum" (eta, gamma): Conjugate GD with learning rate eta and inertia gamma.
    "adagrad" (eta): Adaptive Gradient alogrithm with learning rate eta and simplified diagonal implementation.
    "rmsprop" (eta, beta): Root-Mean-Square-Propagation algorithm with learning rate eta and running average of second moment of gradient weighted with beta.
    "adam" (eta, beta1, beta2): Adam algorithm with learning rate eta, running average of gradient weighted with beta1 and running average of second moment of gradient weighted with beta1. Bias correction of estimates are included.
    """
    def __init__(self, method:str, params:dict, its:int) -> None:
        """Set the type of gradient descent  

        Args:
            method (str): Type of gradient descent 
            params (dict): The hyperparameters for the GD (eta, gamma, betas, etc.)
            its (int): Number of iterations 
        """
        self.method, self.params, self.its = method, params, its
        if method in self.methods.keys():
            init, update = self.methods[method]
            self._initialize = init # set initializing function
            self._update_rule = lambda self, x, grad : update(self, x, grad, **params) # set update rule
        else:
            raise KeyError(f"Method '{method}' not supported, available methods are: " + ", ".join([f"'{method}'" for method in self.methods.keys()]))


    def call(self, grad, x0:np.ndarray, args:tuple=()) -> np.ndarray:
        """Set the problem to be gradient-descended. Create the for-loop with call to method.
        Args:
            grad (callable): The gradient function, returns np.ndarray of same shape as x0 
            x0 (np.ndarray): Starting point
            args (tuple, optional): arguments to be passed to grad-function. Defaults to ().
        """
        # assert that grad works as intended
        grad0 = grad(x0, *args) 
        assert grad0.shape == x0.shape, f"grad-function returns array of shape {grad0.shape} instead of shape {x0.shape}."
        del grad0

        # initialize algorithm
        self._initialize(self, x0)
        # run iterations
        self._it = 0 # tracking iteration for adam/learning schedule
        for it in range(self.its):
            self._it += 1
            g = grad(self.x, *args)
            self.x = self._update_rule(self, self.x, g) 
        print(self.method, self.x)
        return self.x


    def _plain_init(self, x0:np.ndarray) -> None:
        self.x = x0 

    def _plain_update(self, x:np.ndarray, grad:np.ndarray, eta:float) -> np.ndarray:
        return x - eta * grad

    def _momentum_init(self, x0:np.ndarray) -> None:
        self.x = x0
        self.p = np.zeros_like(x0)

    def _momentum_update(self, x:np.ndarray, grad:np.ndarray, eta:float, gamma:float) -> np.ndarray:
        self.p = gamma * self.p + eta * grad 
        return x - self.p

    def _adagrad_init(self, x0:np.ndarray):
        self.x = x0
        self.G = np.zeros_like(x0)

    def _adagrad_update(self, x:np.ndarray, grad:np.ndarray, eta:float, epsilon:float=EPSILON) -> np.ndarray:
        self.G += grad**2
        return x - eta / (np.sqrt(self.G) + epsilon) * grad

    def _rmsprop_init(self, x0:np.ndarray):
        self.x = x0
        self.s = np.zeros_like(x0)

    def _rmsprop_update(self, x:np.ndarray, grad:np.ndarray, eta:float, beta:float, epsilon:float=EPSILON) -> np.ndarray:
        self.s = beta * self.s + (1. - beta) * grad**2
        return x - eta / (np.sqrt(self.s) + epsilon) * grad

    def _adam_init(self, x0:np.ndarray):
        self.x = x0
        self.m = np.zeros_like(x0)
        self.s = np.zeros_like(x0)

    def _adam_update(self, x:np.ndarray, grad:np.ndarray, eta:float, beta1:float, beta2:float, epsilon:float=EPSILON) -> np.ndarray:
        self.m = beta1 * self.m + (1. - beta1) * grad
        self.s = beta2 * self.s + (1. - beta2) * np.square(grad)
        mhat = self.m / (1 - beta1**self._it)
        shat = self.s / (1 - beta2**self._it)
        return x - eta / (np.sqrt(shat) + epsilon) * mhat

    # dict containing the available methods
    methods = {
        "plain": (_plain_init, _plain_update),
        "momentum": (_momentum_init, _momentum_update),
        "adagrad": (_adagrad_init, _adagrad_update),
        "rmsprop": (_rmsprop_init, _rmsprop_update),
        "adam": (_adam_init, _adam_update)
    }


class SGradientDescent(GradientDescent):
    """Implements Gradient Descent minimization of a problem defined by the gradient g of scalar function wrt. argument(s) x. Assumes stochastic form of gradient g(x, idcs) = sum(gi[idcs]) + g0.
    """
    def __init__(self, method:str, params:dict, epochs:int, batch_size:int, random_state=None) -> None:
        """Set the type of gradient descent  

        Args:
            method (str): Type of gradient descent 
            params (dict): The hyperparameters for the GD (eta, gamma, betas, etc.)
            epochs (int): Number of epochs
            batch_size (int): Size of each batch
            random_state (int): random seed to use with numpy.random module 
        """
        if random_state:
            np.random.seed(random_state)
        super().__init__(method, params, epochs)
        self.batch_size = batch_size
        self.epochs = epochs

    def call(self, grad, x0:np.ndarray, all_idcs:np.ndarray, args:tuple=()) -> np.ndarray:
        """Set the problem to be gradient-descended. Create the for-loop with call to method.
        Args:
            grad (callable): The gradient function, returns np.ndarray of same shape as x0.
            x0 (np.ndarray): Starting point.
            all_idcs (np.ndarray): The full set of indices to pass to stochastic gradient function.
            args (tuple, optional): arguments to be passed to grad-function. Defaults to ().
        """
        # assert that grad works as intended
        grad0 = grad(x0, *args) 
        assert grad0.shape == x0.shape, f"grad-function returns array of shape {grad0.shape} instead of shape {x0.shape}."
        del grad0

        # initialize algorithm
        self._initialize(self, x0)
        # run iterations
        self._it = 0 # tracking iteration for adam/learning schedule
        for epoch in range(self.its):
            batches = self._make_batches(all_idcs)
            for batch in batches:
                self._it += 1
                g = grad(self.x, *args, batch)
                self.x = self._update_rule(self, self.x, g) 
        print(self.method, self.x)
        return self.x

    def _make_batches(self, idcs:np.ndarray) -> list:
        """Shuffle idcs and divide into batches with size self.batch_size.

        Args:
            idcs (np.ndarray): Indices to divide into batches.
        """
        np.random.shuffle(idcs) # shuffle indices
        n_batches = len(idcs) // self.batch_size
        return [idcs[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]



if __name__=="__main__":
    from linear_model import OLS_gradient
    from data import Data

    np.random.seed(321)
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = x**2 + np.random.normal(scale=0.1, size=n)
    X = np.c_[np.ones(n), x, x**2]

    # grad = lambda theta, X, y: (2./n) * X.T @ (X @ theta - y)
    theta0 = np.random.randn(X.shape[1]) # makes sure all methods have same starting value
    
    ##################
    # Non-Stochastic #
    ##################
    # GD = GradientDescent(
    #     method = "plain",
    #     params = {"eta":0.8},
    #     its=100,
    # )
    # plain_x = GD.call(
    #     grad=OLS_gradient, 
    #     x0=theta0,
    #     args=(Data(y,X),),
    # )

    # GD_mom = GradientDescent(
    #     method="momentum",
    #     params = {"gamma":0.1, "eta":0.8},
    #     its=100
    # )
    # mom_x = GD_mom.call(
    #     grad=OLS_gradient, 
    #     x0=theta0,
    #     args=(Data(y,X),)
    # )

    # GD_ada = GradientDescent(
    #     method="adagrad",
    #     params = {"eta":0.8},
    #     its=100
    # )
    # ada_x = GD_ada.call(
    #     grad=OLS_gradient, 
    #     x0=theta0,
    #     args=(Data(y,X),)
    # )



    ##############
    # Stochastic #
    ##############

    SGD = SGradientDescent(
        method = "plain",
        params = {"eta":0.1},
        epochs=100,
        batch_size=10
    )
    plain_x = SGD.call(
        grad=OLS_gradient, 
        x0=theta0,
        args=(Data(y,X),),
        all_idcs=np.arange(n)
    )

    SGD_mom = SGradientDescent(
        method="momentum",
        params = {"gamma":0.1, "eta":0.1},
        epochs=100,
        batch_size=10
    )
    mom_x = SGD_mom.call(
        grad=OLS_gradient, 
        x0=theta0,
        args=(Data(y,X),),
        all_idcs=np.arange(n)
    )

    SGD_ada = SGradientDescent(
        method="adagrad",
        params = {"eta":0.1},
        epochs=100,
        batch_size=10
    )
    ada_x = SGD_ada.call(
        grad=OLS_gradient, 
        x0=theta0,
        args=(Data(y,X),),
        all_idcs=np.arange(n)
    )

    analytic_x = np.linalg.pinv(X.T@X) @ X.T@y
    print("Analytic", analytic_x)

    print(f"Plain rel error    : {abs( (plain_x-analytic_x)/analytic_x )}")
    print(f"Momentum rel error : {abs( (mom_x-analytic_x)/analytic_x )}")
    print(f"AdaGrad rel error  : {abs( (ada_x-analytic_x)/analytic_x )}")