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
        self.converged = self.optimizer.converged

        return self


    def fit_save_accuracy(self, data_train:Data, data_test:Data, optimizer, x0:np.ndarray=None):
        self.optimizer = optimizer
        
        if not x0:
            x0 = np.random.normal(0,1, data_train.n_features)
        
        opt_args = {
            "grad": self.grad_save_accuracy,
            "x0": x0,
            "args":(data_train, data_test, self.lmbda)
        }

        if "epochs" in optimizer.__dict__.keys():
            opt_args["all_idcs"] = np.arange(data_train.n_points)

        self.saved_accuracy_opt = []
        self.coef = self.optimizer.call(**opt_args)
        self.saved_accuracy_opt = np.array(self.saved_accuracy_opt)
        self.converged = self.optimizer.converged
        
        return self


    def predict(self, X:np.ndarray, coef=None) -> np.ndarray:
        if coef is None:
            coef = self.coef

        return 1/(1+np.exp( - X @ coef ))


    def classify(self, X:np.ndarray, return_prob:bool=False, coef=None):
        proba = self.predict(X, coef)

        threshold = 0.5

        y_pred = np.where(proba > threshold, 1, 0).astype(int)

        if return_prob:
            return y_pred, proba
        else:
            return y_pred


    def accuracy(self, X:np.ndarray, y:np.ndarray, coef=None):
        n = len(y)
        y_pred = self.classify(X, coef=coef)

        return np.sum(y_pred == y)/n


    def grad(self, coef:np.ndarray, data:Data, lmbda, idcs:np.ndarray=None) -> np.ndarray:
        if idcs is None:
            idcs = np.arange(len(data))
        
        y, X = data[idcs].unpacked()
        n = len(y)

        return (1./n) * X.T @ ( self.predict(X, coef) - y) + lmbda*coef


    def grad_save_accuracy(self, coef:np.ndarray, data_train:Data, data_test:Data, lmbda, idcs:np.ndarray=None) -> np.ndarray:
        self.saved_accuracy_opt.append(self.accuracy(data_test.X, data_test.y, coef))
        
        return self.grad(coef, data_train, lmbda, idcs)