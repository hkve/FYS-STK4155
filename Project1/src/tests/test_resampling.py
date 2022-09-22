import context
import sknotlearn.linear_model
from sknotlearn.resampling import cross_validate

import numpy as np
import sklearn.linear_model
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score



def test_cross_validation():
    np.random.seed(1232)
    x = np.random.uniform(low=-5, high=5, size=10000)
    beta = np.array([1, 0.25, -1.5, 3])

    y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0) + np.random.normal(loc=0, scale=0.5, size=x.shape)
    X = np.c_[np.ones_like(x), x, x**2, x**3]

    k = 10
    kfold = KFold(n_splits=k)
    scores_KFold = np.zeros(k)

    r1 = sknotlearn.linear_model.LinearRegression()
    r2 = sklearn.linear_model.LinearRegression()

    mse1 = cross_val_score(r2, X, y, scoring="neg_mean_squared_error" , cv=kfold)
    mse2 = cross_validate(r1, y, X, k=5, scoring="mse")
    

    tol = 1e-4 # Not necessarily equal fold splitting
    assert np.abs(mse1.mean()+mse2["test_mse"].mean()), f"Discrepancy between sklearn cross validate"