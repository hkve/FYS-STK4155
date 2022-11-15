import classification_utils as cu
from sklearn.preprocessing import StandardScaler
import numpy as np

import context
from sknotlearn.datasets import load_BreastCancer
from sknotlearn.data import Data

def breast_cancer_data(random_state=321):
    D = load_BreastCancer()
    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    _, X_train = D_train.unpacked()
    _, X_test = D_test.unpacked()

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
    
    D_train.X = X_train
    D_test.X = X_test

    return D_train, D_test

def acc_eta_activation_functions(D_train, D_test):
    structures = [
        ((5, ), 1),
        ((30, ), 1),
        ((15, 15), 1),
        ((10, 10, 10), 1)
    ]
    
    eta_ranges = [
        (0.1, 0.4, 8),
        (0.01, 0.1, 8),
        (0.01, 0.1, 8),
        (0.01, 0.1, 8)
    ]

    ylims = [
        (0.95, 1),
        (0.95, 1),
        (0.90, 1),
        (0.90, 1)
    ]

    for i in range(len(structures)):
        nodes = structures[i]
        eta_range = eta_ranges[i]
        ylim = ylims[i]
        filename = f"clasf_activation_functions{i+1}"
       
        cu.varying_activation_functions(D_train, D_test, 
                                activation_functions=["sigmoid", "tanh", "relu", "leaky_relu"],
                                filename=filename,
                                eta_range=eta_range,
                                nodes=nodes,
                                ylim=ylim
                            )


def test_logreg():
    from sklearn.linear_model import LogisticRegression
    D = load_BreastCancer()

    D_train, D_test = D.train_test_split(ratio=3/4, random_state=321)
    clf = LogisticRegression(max_iter=10000).fit(D_train.X, D_train.y)

    y_pred = clf.predict(D_test.X)
    

    acc = np.sum(y_pred == D_test.y)/len(y_pred)
    print(acc)



def lmbda_eta_heatmaps(D_train, D_test):
    eta_range = (0.001, 0.1, 5)
    lmbda_range = (-8, 1, 5)
    activation_hidden="tanh"
    
    structures = [
        ((5,), 1),
        ((10,10,), 1)
    ]

    activations = [
        "sigmoid",
        "tanh"
    ]

    for i, nodes in enumerate(structures):
        for j, af in enumerate(activations):
            filename = f"lmbda_lr_struct{i}_{af}"
            cu.lmbda_eta_heatmap(D_train, D_test, eta_range, lmbda_range, nodes=nodes, activation_hidden=af, filename=filename)

if __name__ == "__main__":
    D_train, D_test = breast_cancer_data()

    acc_eta_activation_functions(D_train, D_test)
    # lmbda_eta_heatmaps(D_train, D_test)