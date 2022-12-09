import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import context
from sknotlearn.datasets import load_fifa
import plot_utils


df = load_fifa(n = 100)

y = df["overall"].to_numpy()
X = df.drop(["overall", "short_name"], axis=1).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = LinearRegression().fit(X_train, y_train)

mse = mean_squared_error(clf.predict(X_test), y_test)
print(mse)
# for y_pred, y in zip(clf.predict(X_test), y_test):
#     print(y_pred, y)