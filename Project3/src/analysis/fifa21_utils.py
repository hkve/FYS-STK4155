import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from mlxtend.evaluate import bias_variance_decomp

import context
from sknotlearn.datasets import load_fifa

class LinearRegressionInt(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)

class RidgeInt(Ridge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)

class LassoInt(Lasso):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)

class DecisionTreeRegressorInt(DecisionTreeRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)

    def fit(self, X, y, **kwargs):
        tmp = super().fit(X, y, **kwargs)
        mse = np.mean((self.predict(X)-y)**2)
        self.train_mse = mse

        return tmp
class BaggingRegressorInt(BaggingRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)

class GradientBoostingRegressorInt(GradientBoostingRegressor):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)

    def predict(self, X):
        return np.round_(super().predict(X), decimals=0).astype(int)


class CustomScaler(StandardScaler): 
    def __init__(self):
        self.n_ohe = 5
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X[:, :-self.n_ohe])
        return self

    def transform(self, X):
        X_head = self.scaler.transform(X[:, :-self.n_ohe])
        return np.c_[X_head, X[:, -self.n_ohe:]]
    

def get_fifa_data(n=100, random_state=321):
    df = load_fifa(n = 100, random_state=random_state)
    
    int_reps = pd.Index(["1","2","3","4","5"])
    int_rep_ohe = pd.get_dummies(df["international_reputation"], prefix='', prefix_sep='').reindex(columns = int_reps, fill_value=0)

    y = df["overall"].to_numpy()
    X = df.drop(["overall", "short_name", "international_reputation"], axis=1).to_numpy()

    X = np.c_[X, int_rep_ohe]

    return X, y

def process_fifa_data(X, y, tts=3/4, random_state=321):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=tts, random_state=321)
    scaler = CustomScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def bootstrap(X, y, method, param_name, params, method_params=None, save_regs=False, random_state=321):
    n = len(params)
    mse, bias, var = np.zeros(n), np.zeros(n), np.zeros(n)
    X_train, X_test, y_train, y_test = process_fifa_data(X, y, random_state=random_state)

    regs = []
    for i, param in enumerate(params):
        if method_params:
            reg = method(**{param_name: param}, **method_params)
        else:
            reg = method(**{param_name: param})

        mse[i], bias[i], var[i] = bias_variance_decomp(reg, X_train, y_train, X_test, y_test, loss="mse", num_rounds=200, random_seed=random_state)

        if save_regs:
            regs.append(reg)

    if save_regs:
        return mse, bias, var, regs
    else:
        return mse, bias, var

def boostrap_single(X, y, method, method_params=None, random_state=321):
    X_train, X_test, y_train, y_test = process_fifa_data(X, y, random_state=random_state)

    reg = method(**method_params)

    return bias_variance_decomp(reg, X_train, y_train, X_test, y_test, loss="mse", num_rounds=200, random_seed=random_state)

def bv_decomp(X, y, reg, random_state=321):
    X_train, X_test, y_train, y_test = process_fifa_data(X, y, random_state=random_state)

    return bias_variance_decomp(reg, X_train, y_train, X_test, y_test, loss="mse", num_rounds=200, random_seed=random_state)