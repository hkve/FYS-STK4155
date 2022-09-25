from random import random
import context
from sknotlearn.linear_model import LinearRegression
from sknotlearn.resampling import cross_validation
from sknotlearn.datasets import make_FrankeFunction
from sknotlearn.data import Data
import numpy as np

def task_d(n=600, ks=np.arange(5,10+1), random_state=321):
    mse, r2 = np.zeros_like(ks), np.zeros_like(ks)
    
    X, y = make_FrankeFunction(n, noise_std=0.1, random_state=random_state)


if __name__ == "__main__":
    task_d()