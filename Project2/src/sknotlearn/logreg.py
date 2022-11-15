import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from data import Data
from optimize import GradientDescent, SGradientDescent

class LogisticRegression:
    def __init__(self, lmbda=0):
        self.lmbda = lmbda

    def fit(self, data:Data, optimizer, x0:np.ndarray=None):
        self.optimizer = optimizer
        
        if not x0:
            x0 = np.random.normal(0,1, data.n_features)
        
        opt_args = {
            "grad": self.grad,
            "x0": x0,
            "args":(data, self.lmbda)
        }

        if "epochs" in optimizer.__dict__.keys():
            opt_args["all_idcs"] = np.arange(data.n_points)

        self.coef = self.optimizer.call(**opt_args)

        return self

    def predict(self, X:np.ndarray, coef=None) -> np.ndarray:
        if coef is None:
            coef = self.coef

        return 1/(1+np.exp( - X @ coef ))

    def classify(self, X:np.ndarray, return_prob:bool=False):
        proba = self.predict(X)
        threshold = 0.5

        y_pred = np.where(proba > threshold, 1, 0).astype(int)

        if return_prob:
            return y_pred, proba
        else:
            return y_pred

    def accuracy(self, X:np.ndarray, y:np.ndarray):
        n = len(y)
        y_pred = self.classify(X)
        return np.sum(y_pred == y)/n

    def grad(self, coef:np.ndarray, data:Data, lmbda, idcs:np.ndarray=None) -> np.ndarray:
        if idcs is None:
            idcs = np.arange(len(data))
        
        y, X = data[idcs].unpacked()
        n = len(y)

        return (1./n) * X.T @ ( self.predict(X, coef) - y) + lmbda*coef