import numpy as np

class GradientDescent:
    """_summary_
    """
    def __init__(self, method:str, params:dict[float], its:int) -> None:
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


    def call(self, grad:callable, x0:np.ndarray, args:tuple=()) -> np.ndarray:
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
        for it in range(self.its):
            # run iterations
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
        self.p = 0

    def _momentum_update(self, x:np.ndarray, grad:np.ndarray, eta:float, gamma:float) -> np.ndarray:
        self.p = gamma * self.p + eta * grad 
        return x - self.p

    # dict containing the available methods
    methods = {
        "plain": (_plain_init, _plain_update),
        "momentum": (_momentum_init, _momentum_update)
    }


class SGradientDescent(GradientDescent):
    pass



if __name__=="__main__":
    np.random.seed(321)
    n = 100
    x = np.random.uniform(-1, 1, n)
    y = x**2 + np.random.normal(scale = 0.1, size=n)
    X = np.c_[np.ones(n), x, x**2]
    grad = lambda theta, X, y: (2./n)*X.T@(X@theta - y)
    GD = GradientDescent(
        method = "plain",
        params = {"eta":0.8},
        its=100
    )
    GD.call(
        grad=grad, 
        x0=np.random.randn(3), 
        args=(X,y)
    )

    GD_mom = GradientDescent(
        method="momentum",
        params = {"gamma":0.1, "eta":0.8},
        its=100
    )
    GD_mom.call(
        grad=grad, 
        x0=np.random.randn(3), 
        args=(X,y)
    )
    
    print("Analytic", np.linalg.pinv(X.T@X) @ X.T@y)
