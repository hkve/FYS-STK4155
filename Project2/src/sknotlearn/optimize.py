import numpy as np

class GradientDescent:
    """_summary_
    """
    def __init__(self, method, params, its):
        """Set the type of gradient descent  

        Args:
            method (_type_): Type of gradient descent 
            params (dict): The hyperparameters for the GD (eta, gamma, betas, etc.)
            its (int): Number of iterations 
        """
        self.method, self.params, self.its = method, params, its
        if method in self.methods.keys():
            init, update = self.methods[method]
            self._initialize = init
            self._update_rule = lambda self, x, grad : update(self, x, grad, **params)
        

    def call(self, grad, x0, args=()):
        """Set the problem to be gradient-descended. Create the for-loop with call to method.
        Args:
            cost_func (_type_): The cost function 
            grad (): 
            x0 (np.ndarray): Starting point
            args (tuple, optional): _description_. Defaults to ().
        """
        self._initialize(self, x0)
        for it in range(self.its):
            g = grad(self.x, *args)
            self.x = self._update_rule(self, self.x, g) 
        print(self.method, self.x)


    def plain_init(self, x0):
        self.x = x0 

    def plain_update(self, x, grad, eta):
        return x - eta * grad

    def momentum_init(self, x0):
        self.x = x0
        self.p = 0

    def momentum_update(self, x, grad, eta, gamma):
        self.p = gamma * self.p + eta * grad 
        return x - self.p

    methods = {
        "plain": (plain_init, plain_update),
        "momentum": (momentum_init, momentum_update)
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
