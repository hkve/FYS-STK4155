import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plot_utils
from utils import make_figs_path

import context
from sknotlearn.data import Data
from sknotlearn.datasets import load_BreastCancer
from sknotlearn.optimize import SGradientDescent
from sknotlearn.neuralnet import NeuralNetwork

def main():
    D = load_BreastCancer()

    SGD = SGradientDescent(
        method = "adam",
        params = {"eta":0.1, "beta1": 0.9, "beta2": 0.99},
        epochs=100,
        batch_size=20,
        random_state=321
    )

    nodes = ((20, ), 1)
    NN = NeuralNetwork(
        SGD, 
        nodes=nodes, 
        random_state=321,
        cost_func="BCE",
        activation_hidden="sigmoid",
        activation_output="sigmoid"
    )

    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    y_train, X_train = D_train.unpacked()
    y_test, X_test = D_test.unpacked()

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)


    D_train.X = X_train
    NN.train(D_train, trainsize=1)
    print(NN.accuracy(X_test, y_test))
    

def test_logreg():
    from sklearn.linear_model import LogisticRegression
    D = load_BreastCancer()

    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    clf = LogisticRegression(max_iter=10000).fit(D_train.X, D_train.y)

    y_pred = clf.predict(D_test.X)

    acc = np.sum(y_pred == D_test.y)/len(y_pred)
    print(acc)


if __name__ == "__main__":
    test_logreg()
    main()