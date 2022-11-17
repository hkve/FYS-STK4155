import numpy as np
from sys import float_info, exit
from collections.abc import Callable
EPSILON = float_info.epsilon**0.5
MAX = 1 / float_info.epsilon


def isNotFinite(x: np.ndarray, threshold: float = MAX):
    return any(np.abs(x) > threshold) or any(np.isnan(x))


class GradientDescent:
    """Implements Gradient Descent minimization of a problem defined by the gradient g of scalar function wrt. argument(s) x. Implemented update rules are:
    "plain" (eta): Ordinary GD with a learning rate eta.
    "momentum" (eta, gamma): Conjugate GD with learning rate eta and inertia gamma.
    "adagrad" (eta): Adaptive Gradient alogrithm with learning rate eta and simplified diagonal implementation.
    "rmsprop" (eta, beta): Root-Mean-Square-Propagation algorithm with learning rate eta and running average of second moment of gradient weighted with beta.
    "adam" (eta, beta1, beta2): Adam algorithm with learning rate eta, running average of gradient weighted with beta1 and running average of second moment of gradient weighted with beta1. Bias correction of estimates are included.
    """

    def __init__(self, method: str, params: dict, its: int, threshold: float = MAX) -> None:
        """Set the type of gradient descent

        Args:
            method (str): Type of gradient descent
            params (dict): The hyperparameters for the GD (eta, gamma, betas, etc.)
            its (int): Number of iterations
            threshold (float): Maximum float value for gradients. Defaults to MAX.
        """
        self.method, self.params, self.its = method, params, its
        self.threshold = threshold
        if method in self.methods.keys():
            if not callable(params["eta"]):  # wrap learning rate if constant
                eta = params["eta"]
                params["eta"] = lambda it: eta
            init, update = self.methods[method]
            self._initialize = init  # set initializing function
            self._update_rule = lambda self, x, grad: update(
                self, x, grad, **params)  # set update rule
        else:
            raise KeyError(f"Method '{method}' not supported, available methods are: "
                           + ", ".join([f"'{method}'" for method in self.methods.keys()]))

    def set_params(self, params: dict) -> None:
        if not callable(params["eta"]):  # wrap learning rate if constant
            eta = params["eta"]
            params["eta"] = lambda it: eta
        self.params = params
        init, update = self.methods[self.method]
        self._initialize = init
        self._update_rule = lambda self, x, grad: update(
            self, x, grad, **params)

    def call(self, grad, x0: np.ndarray, args: tuple = ()) -> np.ndarray:
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
        self._it = 0  # tracking iteration for adam/learning schedule
        for it in range(self.its):
            self._it += 1
            g = grad(self.x, *args)
            if isNotFinite(np.abs(g), threshold=self.threshold):
                self.converged = 0
                return self.x
            self.x = self._update_rule(self, self.x, g)
        # print(self.method, self.x)
        self.converged = 1
        return self.x

    def _plain_init(self, x0: np.ndarray) -> None:
        self.x = x0

    def _plain_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable) -> np.ndarray:
        return x - eta(self._it) * grad

    def _momentum_init(self, x0: np.ndarray) -> None:
        self.x = x0
        self.p = np.zeros_like(x0)

    def _momentum_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable, gamma: float) -> np.ndarray:
        self.p = gamma * self.p + eta(self._it) * grad
        return x - self.p

    def _adagrad_init(self, x0: np.ndarray):
        self.x = x0
        self.G = np.zeros_like(x0)

    def _adagrad_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable, epsilon: float = EPSILON) -> np.ndarray:
        self.G += grad**2
        return x - eta(self._it) / (np.sqrt(self.G) + epsilon) * grad

    def _adagrad_momentum_init(self, x0: np.ndarray):
        self.x = x0
        self.p = np.zeros_like(x0)
        self.G = np.zeros_like(x0)

    def _adagrad_momentum_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable, gamma: float, epsilon: float = EPSILON) -> np.ndarray:
        self.G += grad**2
        self.p = gamma * self.p + eta(self._it) * grad
        return x - self.p / (np.sqrt(self.G) + epsilon)

    def _rmsprop_init(self, x0: np.ndarray):
        self.x = x0
        self.s = np.zeros_like(x0)

    def _rmsprop_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable, beta: float, epsilon: float = EPSILON) -> np.ndarray:
        self.s = beta * self.s + (1. - beta) * grad**2
        return x - eta(self._it) / (np.sqrt(self.s) + epsilon) * grad

    def _adam_init(self, x0: np.ndarray):
        self.x = x0
        self.m = np.zeros_like(x0)
        self.s = np.zeros_like(x0)

    def _adam_update(self, x: np.ndarray, grad: np.ndarray, eta: Callable, beta1: float, beta2: float, epsilon: float = EPSILON) -> np.ndarray:
        self.m = beta1 * self.m + (1. - beta1) * grad
        self.s = beta2 * self.s + (1. - beta2) * np.square(grad)
        mhat = self.m / (1 - beta1**self._it)
        shat = self.s / (1 - beta2**self._it)
        return x - eta(self._it) / (np.sqrt(shat) + epsilon) * mhat

    # dict containing the available methods
    methods = {
        "plain": (_plain_init, _plain_update),
        "momentum": (_momentum_init, _momentum_update),
        "adagrad": (_adagrad_init, _adagrad_update),
        "adagrad_momentum": (_adagrad_momentum_init, _adagrad_momentum_update),
        "rmsprop": (_rmsprop_init, _rmsprop_update),
        "adam": (_adam_init, _adam_update)
    }


class SGradientDescent(GradientDescent):
    """Implements Gradient Descent minimization of a problem defined by the gradient g of scalar function wrt. argument(s) x. Assumes stochastic form of gradient g(x, idcs) = sum(gi[idcs]) + g0.
    """

    def __init__(self, method: str, params: dict, epochs: int, batch_size: int, random_state=None, threshold: float = MAX) -> None:
        """Set the type of gradient descent

        Args:
            method (str): Type of gradient descent
            params (dict): The hyperparameters for the GD (eta, gamma, betas, etc.)
            epochs (int): Number of epochs
            batch_size (int): Size of each batch
            random_state (int): random seed to use with numpy.random module
        """
        super().__init__(method, params, epochs, threshold)
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

    def call(self, grad, x0: np.ndarray, all_idcs: np.ndarray, args: tuple = ()) -> np.ndarray:
        """Set the problem to be gradient-descended. Create the for-loop with call to method.
        Args:
            grad (callable): The gradient function, returns np.ndarray of same shape as x0.
            x0 (np.ndarray): Starting point.
            all_idcs (np.ndarray): The full set of indices to pass to stochastic gradient function.
            args (tuple, optional): arguments to be passed to grad-function. Defaults to ().
        """
        # random_state is set, every call will use the same batches
        if self.random_state:
            np.random.seed(self.random_state)
        # assert that grad works as intended
        grad0 = grad(x0, *args, all_idcs)
        assert grad0.shape == x0.shape, f"grad-function returns array of shape {grad0.shape} instead of shape {x0.shape}."
        del grad0

        # initialize algorithm
        self._initialize(self, x0)
        # run iterations
        self._it = 0  # tracking iteration for adam/learning schedule
        for epoch in range(self.its):
            batches = self._make_batches(all_idcs)
            self._it += 1
            for batch in batches:
                g = grad(self.x, *args, batch)
                if isNotFinite(np.abs(g), threshold=self.threshold):
                    self.converged = 0
                    return self.x
                self.x = self._update_rule(self, self.x, g)
        # print(self.method, self.x)
        self.converged = 1
        return self.x

    def _make_batches(self, idcs: np.ndarray) -> list:
        """Shuffle idcs and divide into batches with size self.batch_size.

        Args:
            idcs (np.ndarray): Indices to divide into batches.
        """
        np.random.shuffle(idcs)  # shuffle indices
        n = len(idcs)
        n_batches = n // self.batch_size
        if n_batches == 0:  # If batch size is larger than n, we use 1 batch
            n_batches = 1
        return [idcs[i:i+n//n_batches] for i in range(n_batches)]


if __name__ == "__main__":
    from linear_model import OLS_gradient
    from datasets import make_FrankeFunction
    random_state = 321
    np.random.seed(random_state)
    D = make_FrankeFunction(n=600, noise_std=0.1, random_state=random_state)
    D = D.polynomial(degree=5)
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=random_state)
    D_train = D_train.scaled("Standard")
    D_test = D_train.scale(D_test)

    max_iter = 10000
    theta0 = np.random.randn(D.n_features)

    GD = GradientDescent("momentum", {"eta": 0.10, "gamma": 0.8},
                         its=max_iter)
    SGD = SGradientDescent("momentum", {"eta": 0.10, "gamma": 0.8},
                           epochs=max_iter, batch_size=128)
    theta_GD = GD.call(
        grad=OLS_gradient,
        x0=theta0,
        args=(D_train,)
    )
    theta_SGD = SGD.call(
        grad=OLS_gradient,
        x0=theta0,
        all_idcs=np.arange(len(D_train)),
        args=(D_train,)
    )

    theta_ana = np.linalg.pinv(D_train.X.T@D_train.X) @ D_train.X.T @ D_train.y

    MSE_GD = np.mean((D_test.X@theta_GD - D_test.y)**2)
    MSE_SGD = np.mean((D_test.X@theta_SGD - D_test.y)**2)
    MSE_ana = np.mean((D_test.X@theta_ana - D_test.y)**2)

    print(MSE_GD, MSE_SGD, MSE_ana)
