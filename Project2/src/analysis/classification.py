import classification_utils as cu
from sklearn.preprocessing import StandardScaler
import numpy as np

import context
from sknotlearn.datasets import load_BreastCancer
from sknotlearn.data import Data
from sknotlearn.optimize import GradientDescent, SGradientDescent

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
        ((10, ), 1), # 10
        ((5, 5), 1),
        ((5, 5, 5), 1)
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
        filename = None
        
        cu.varying_activation_functions(D_train, D_test, 
                                activation_functions=["sigmoid", "tanh", "relu", "leaky_relu"],
                                filename=filename,
                                eta_range=eta_range,
                                nodes=nodes,
                                ylim=ylim
                            )



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

def logreg_different_opts(D_train, D_test):
    eta_range = (.1,1, 20)

    GD_few = GradientDescent("plain", dict(eta=0.), its=5)
    GD_many = GradientDescent("plain", dict(eta=0.), its=50)
    mGD_few = GradientDescent("momentum", dict(gamma=0.8, eta=0.), its=5)
    mGD_many = GradientDescent("momentum", dict(gamma=0.8, eta=0.), its=50)

    labels = ["GD, 5 iters", "GD, 50 iters", "mGD, 5 iters", "mGD, 50 iters"]
    opts = [GD_few, GD_many, mGD_few, mGD_many]    
    cu.logreg_different_opts(D_train, D_test, eta_range, opts, labels, filename="logreg_GD")


    SGD_few = SGradientDescent("plain", dict(eta=0.), epochs=5, batch_size=200)
    SGD_many = SGradientDescent("plain", dict(eta=0.), epochs=50, batch_size=200)
    AdamSGD = SGradientDescent("adam", dict(eta=0., beta1=0.9, beta2=0.99), epochs=5, batch_size=200)
    adSGD = SGradientDescent("adagrad", dict(eta=0.), epochs=5, batch_size=200)

    labels = ["SGD, 5 epochs", "SGD2, 50 epochs", "Adam SGD, 5 epochs", "AdaGrad SGD, 5 epochs"]
    opts = [SGD_few, SGD_many, AdamSGD, adSGD] # , mGD, SGD, adSGD

    cu.logreg_different_opts(D_train, D_test, eta_range, opts, labels, filename="logreg_SGD")

def logreg_different_opts_during_training(D_train, D_test):
    max_its = 200
    eta = 0.05
    its = np.arange(1, max_its+1)
    GD1 = GradientDescent("plain", dict(eta=eta), its=max_its)
    mGD1 = GradientDescent("momentum", dict(gamma=0.8, eta=eta), its=max_its)
    SGD = SGradientDescent("plain", dict(eta=eta), epochs=max_its, batch_size=200)
    AdamSGD = SGradientDescent("adam", dict(eta=eta, beta1=0.9, beta2=0.99), epochs=max_its, batch_size=200)
    adSGD = SGradientDescent("adagrad", dict(eta=eta), epochs=max_its, batch_size=200)


    opts = [GD1, mGD1, SGD, AdamSGD, adSGD]
    labels = ["GD", "mGD", "SGD", "Adam SGB", "AdaGrad SGD"]

    cu.logreg_different_opts_during_opts(D_train, D_test, its, opts, labels, filename="logreg_acc_during_optimazation")


if __name__ == "__main__":
    D_train, D_test = breast_cancer_data()
    # acc_eta_activation_functions(D_train, D_test)
    # lmbda_eta_heatmaps(D_train, D_test)

    # cu.lmbda_eta_heatmap_sklearn(D_train, D_test)
    # cu.logreg_with_sklearn(D_train, D_test)
    # logreg_different_opts(D_train, D_test)
    # logreg_different_opts_during_training(D_train, D_test)
    # cu.lmbda_eta_heatmap_sklearn(D_train, D_test, (0.001, 0.5, 5), lmbdas=(-6,0,5), filename="lmbda_lr_struct1_sigmoid_sklearn")

    # cu.logreg_heatmap(D_train, D_test, etas=(0.001, 0.5, 5), lmbdas=(-6,0,5), filename="logreg_penalisation")
    cu.logreg_with_sklearn(D_train, D_test, (0.01, 3, 40), filename="logreg_penalisation_sklearn")