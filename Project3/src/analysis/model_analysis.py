"""Contains plotting functions for comparing model performances."""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd


def plot_history(history, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    pd.DataFrame(history).plot(ax=ax)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")

    return ax


def fit_and_plot_val_accuracy(models: list[tf.keras.Model],
                              model_names: list[str],
                              fit_kwargs: dict(),
                              ax=None,
                              filename=None):
    if ax is None:
        _, ax = plt.subplots()

    for model, model_name in zip(models, model_names):
        results = model.fit(**fit_kwargs)
        ax.plot(results.history["val_accuracy"], label=model_name)
        print(f"Best accuracy of {model_name} : "
              f"{np.max(results.history['val_accuracy'])}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.legend()

    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))

    return ax


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    import plot_utils

    import context
    from sknotlearn.datasets import load_MNIST, load_CIFAR10, load_EPL
    from tensorno.bob import build_LWTA_classifier, build_FFNN_classifier
    from tensorno.utils import count_parameters

    # x_train, y_train, x_test, y_test = load_MNIST()

    x_train, y_train, x_test, y_test = load_CIFAR10()

    # Flattening image arrays.
    x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    # dataset = load_EPL(encoded=True)
    # y = dataset.y.to_numpy()
    # labels = np.array(["w", "d", "l"])
    # y = np.array([np.where(y_ == range(len(labels)), 1, 0) for y_ in y],
    #              dtype=int)
    # x = dataset.x
    # x = x.astype(float)
    # x_train, x_test, y_train, y_test = train_test_split(x, y,
    #                                                     train_size=5/6,
    #                                                     shuffle=False)

    # Scaling data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    """Saving some good models in a bad way..."""
    # Maxout on CIFAR10 data
    model1 = build_LWTA_classifier(  # CIFAR10
        num_layers=3,
        units=[64, 32, 64],
        num_groups=[32, 16, 8],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        Layer="max_out",
    )
    model2 = build_LWTA_classifier(  # CIFAR10_l2
        num_layers=2,
        units=[64, 32],
        num_groups=[32, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        lmbda=1e-4,
        Layer="max_out",
    )
    model3 = build_LWTA_classifier(  # CIFAR10_do
        num_layers=2,
        units=[64, 64],
        num_groups=[32, 16],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.1,
        Layer="max_out",
    )
    model4 = build_LWTA_classifier(  # CIFAR10_do_l2
        num_layers=2,
        units=[32, 64],
        num_groups=[16, 16],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.1,
        lmbda=1e-4,
        Layer="max_out",
    )

    # Channel-out on CIFAR-10 data
    model1 = build_LWTA_classifier(  # CIFAR10
        num_layers=3,
        units=[64, 16, 32],
        num_groups=[16, 8, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        Layer="channel_out",
    )
    model2 = build_LWTA_classifier(  # CIFAR10_l2
        num_layers=3,
        units=[64, 16, 64],
        num_groups=[32, 16, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        lmbda=1e-4,
        Layer="channel_out",
    )
    model3 = build_LWTA_classifier(  # CIFAR10_do
        num_layers=3,
        units=[32, 16, 64],
        num_groups=[16, 16, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.1,
        Layer="channel_out",
    )
    model4 = build_LWTA_classifier(  # CIFAR10_dp_l2
        num_layers=2,
        units=[32, 64],
        num_groups=[16, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.5,
        lmbda=1e-4,
        Layer="channel_out",
    )

    # LWTA on EPL data
    model1 = build_LWTA_classifier(  # EPL
        num_layers=2,
        units=[8, 16],
        num_groups=[8, 8],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        Layer="max_out",
    )
    model2 = build_LWTA_classifier(  # EPL_do_l2
        num_layers=2,
        units=[16, 64],
        num_groups=[8, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.25,
        lmbda=1e-4,
        Layer="max_out",
    )
    model3 = build_LWTA_classifier(  # EPL
        num_layers=4,
        units=[8, 64, 16, 64],
        num_groups=[8, 32, 16, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        Layer="channel_out",
    )
    model4 = build_LWTA_classifier(  # EPL_dp_l2
        num_layers=2,
        units=[8, 64],
        num_groups=[8, 32],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=0.25,
        lmbda=1e-4,
        Layer="channel_out",
    )

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=30,
                                                     verbose=1,
                                                     restore_best_weights=True)

    fit_and_plot_val_accuracy(
        models=[model1, model2, model3, model4],
        # model_names=[r"channelout", r"channelout w/ L2",
        #              r"channelout w/ DO", r"channelout w/ DO \& L2"],
        model_names=[r"maxout", r"maxout w/ DO \& L2",
                     r"channelout", r"channelout w/ DO \& L2"],
        fit_kwargs=dict(
            x=x_train,
            y=y_train,
            epochs=300,
            validation_data=(x_test, y_test),
            callbacks=[early_stopper]
        ),
        filename="best_models_EPL"
    )
    plt.show()

    """
    results = model2.fit(
        x_train, y_train,
        epochs=300,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper]
    )

    model2.evaluate(x_test, y_test)
    print(f"Number of parameters in network: {count_parameters(model2)}")

    history = {
        "train accuracy NN": results.history["accuracy"],
        "test accuracy NN": results.history["val_accuracy"]
    }
    _, ax = plt.subplots()
    ax = plot_history(history, ax=ax)

    ###
    # Ordinary Network
    ###
    model = build_FFNN_classifier(  # Dense model
        num_layers=2,
        units=[16, 16],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        lmbda=1e-4,
        activation="ReLU",
    )

    results = model.fit(
        x_train, y_train,
        epochs=300,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper, ]
    )

    model.evaluate(x_test, y_test)
    print(f"Number of parameters in network: {count_parameters(model)}")

    history = {
        "train accuracy NN": results.history["accuracy"],
        "test accuracy NN": results.history["val_accuracy"]
    }
    plot_history(history)

    plt.show()
    """
