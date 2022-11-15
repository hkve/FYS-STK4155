import classification_utils as cu
from sklearn.preprocessing import StandardScaler

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
        ((1, ), 1),
        ((30, ), 1),
        ((15, 15), 1),
        ((10, 10, 10), 1)
    ]
    
    eta_ranges = [
        (0.01, 0.4, 5),
        (0.001, 0.1, 5),
        (0.001, 0.1, 5),
        (0.001, 0.1, 5)
    ]

    ylims = [
        (0.93, 1),
        (0.88, 1),
        (0.9, 1),
        (0.85, 1)
    ]

    for i in range(len(structures)):
        nodes = structures[i]
        eta_range = eta_ranges[i]
        filename = f"clasf_activation_functions{i+1}"
        filename=None
       
        cu.varying_activation_functions(D_train, D_test, 
                                activation_functions=["sigmoid", "tanh", "relu", "leaky_relu", "linear"],
                                filename=filename,
                                eta_range=eta_range,
                                nodes=nodes,
                            )


def lmbda_eta_heatmaps(D_train, D_test):
    eta_range = (-5, -1, 5)
    lmbda_range = (-8, 1, 5)

    cu.lmbda_eta_heatmap(D_train, D_test, eta_range, lmbda_range, nodes=((1,), 1))

if __name__ == "__main__":
    D_train, D_test = breast_cancer_data()

    # lmbda_eta_heatmaps(D_train, D_test)
    acc_eta_activation_functions(D_train, D_test)