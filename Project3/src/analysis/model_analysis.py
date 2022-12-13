"""Contains plotting functions for comparing model performances."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def plot_loss_history(history, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    pd.DataFrame(history).plot(ax=ax)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    import plot_utils

    import context
    from sknotlearn.datasets import load_MNIST, load_CIFAR10
    from tensorno.bob import build_LWTA_classifier, build_FFNN_classifier
    from tensorno.utils import count_parameters

    # x_train, y_train, x_test, y_test = load_MNIST()
    x_train, y_train, x_test, y_test = load_CIFAR10()

    # Flattening image arrays.
    x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    """Saving some good models in a bad way..."""
    model = build_LWTA_classifier(  # CIFAR10_1
        num_layers=2,
        units=[32, 64],
        num_groups=[16, 16],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.1,
        lmbda=1e-4,
        Layer="max_out",
    )
    # model = build_LWTA_classifier(  # CIFAR10_1
    #     num_layers=2,
    #     units=[64, 16],
    #     num_groups=[8, 8],
    #     num_features=x_train.shape[-1],
    #     num_categories=y_train.shape[-1],
    #     dropout_rate=0.1,
    #     lmbda=1e-4,
    #     Layer="channel_out",
    # )
    # model = build_LWTA_classifier(  # CIFAR10_2
    #     num_layers=3,
    #     units=[64, 16, 8],
    #     num_groups=[8, 16, 4],
    #     num_features=x_train.shape[-1],
    #     num_categories=y_train.shape[-1],
    #     dropout_rate=0.1,
    #     lmbda=1e-4,
    #     Layer="max_out",
    # )
    # model = build_LWTA_classifier(  # MNIST_1
    #     num_layers=3,
    #     units=[16, 16, 32],
    #     num_groups=[8, 16, 16],
    #     num_features=x_train.shape[-1],
    #     num_categories=y_train.shape[-1],
    #     dropout_rate=0.1,
    #     lmbda=1e-4,
    #     Layer="max_out",
    # )

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=5,
                                                     verbose=1,
                                                     restore_best_weights=True)

    results = model.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper]
    )

    model.evaluate(x_test, y_test)
    print(f"Number of parameters in network: {count_parameters(model)}")

    _, ax = plt.subplots()

    history = {
        "train loss LWTA": results.history["loss"],
        "val loss LWTA": results.history["val_loss"]
    }
    plot_loss_history(history, ax=ax)

    ###
    # Ordinary Network
    ###
    model = build_FFNN_classifier(  # Dense model
        num_layers=2,
        units=[32, 64],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        lmbda=1e-4,
        activation="ReLU",
    )

    results = model.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper]
    )

    model.evaluate(x_test, y_test)
    print(f"Number of parameters in network: {count_parameters(model)}")

    history = {
        "train loss NN": results.history["loss"],
        "val loss NN": results.history["val_loss"]
    }
    plot_loss_history(history, ax=ax)

    plt.show()
