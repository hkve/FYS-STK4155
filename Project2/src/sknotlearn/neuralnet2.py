import numpy as np
from sys import exit

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x) * sigmoid(x)**2

def MSE(y, y_pred):
    return np.mean((y_pred-y)**2)

def dMSE(y, y_pred):
    return (2./len(y)) * (y_pred-y)

def nn_regressor(
    inputs,
    targets,
    nodes,
    eta,
    epochs,
    n_batches,
    iw_h=None,
    ib_h=None,
    iw_o=None,
    ib_o=None,
    random_state=None
):
    if random_state:
        np.random.seed(random_state)

    # initialize weights, biases
    if iw_h is None:
        iw_h = np.random.randn(1,nodes)
    if ib_h is None:
        ib_h = 0.1 * np.ones(nodes)
    if iw_o is None:
        iw_o = np.random.randn(nodes,1)
    if ib_o is None:
        ib_o = 0.1 * np.ones(1)
    w_h, b_h, w_o, b_o = iw_h, ib_h, iw_o, ib_o

    n = len(inputs)
    idcs = np.arange(n)
    counter = 0
    for epoch in range(epochs):
        np.random.shuffle(idcs)
        batches = [idcs[i:i+n//n_batches] for i in range(n_batches)]
        for batch in batches:
            counter += 1
            # forward pass
            z_h = inputs[batch] @ w_h + b_h
            a_h = sigmoid(z_h)
            output = a_h @ w_o + b_o


            # backpropagate
            delta_o = dMSE(targets[batch], output)
            delta_h = delta_o @ w_o.T * dsigmoid(z_h)

            # if counter == 3200:
            #     print(w_h)
            #     exit()
            # gradient descent step
            w_h = w_h - eta * inputs[batch].T @ delta_h
            b_h = b_h - eta * delta_h.sum(axis=0)
            w_o = w_o - eta * a_h.T @ delta_o
            b_o = b_o - eta * delta_o.sum(axis=0)
        predictions = sigmoid(inputs@w_h + b_h) @ w_o + b_o
        # print(MSE(targets, predictions))

    print(np.mean((predictions-targets)**2))
    exit()
    return predictions

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from datasets import make_debugdata
    from data import Data

    x, y, X = make_debugdata(n=256, random_state=123)
    x = (x - np.mean(x)) / np.std(x)
    y, X = Data(y, X).scaled(scheme="Standard").unpacked()

    y_pred = nn_regressor(
        inputs=x.reshape(-1,1),
        targets=y.reshape(-1,1),
        nodes=10,
        eta=0.06,
        epochs=100,
        n_batches=32,
        random_state=321
    )

    sorted_idcs = x.argsort()

    theta_OLS = np.linalg.pinv(X.T@X) @ X.T @ y
    print(np.mean((X@theta_OLS-y)**2))

    plt.scatter(x, y, c="r")
    plt.plot(x[sorted_idcs], y_pred[sorted_idcs])
    plt.plot(x[sorted_idcs], (X@theta_OLS)[sorted_idcs], ls="--")
    plt.show()