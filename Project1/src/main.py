from sknotlearn.linear_model import LinearRegression as LR_custom
from sklearn.linear_model import LinearRegression as LR

import numpy as np

# Set 50 datapoints and 4 features (including intercept)
n, m = 50, 4
x = np.random.uniform(low=-5, high=5, size=n)

# The 4 true parameters and model y = sum( beta_i * x^i ) 
beta = np.array([1, 0.25, -1.5, 3])
y = np.sum(np.array([b*x**i for i, b in enumerate(beta)]), axis=0)

# Construct design matrix
one = np.ones_like(x)
X = np.c_[one, x, x**2, x**3]

# Compeare between sklearn and custom code
lin1 = LR(fit_intercept=False).fit(X, y)

# Method can be INV or SVD 
lin2 = LR_custom(method="INV").fit(X, y)
lin3 = LR_custom(method="SVD").fit(X, y)

print(lin1.coef_)
print(lin2.coef_)
print(lin3.coef_)

