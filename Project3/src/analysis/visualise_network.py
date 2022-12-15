"""Contains the code used to plot visualise LWTA NN architecture
and activation."""
import matplotlib.pyplot as plt
import tensorflow as tf

import network_plot_tools as npt
import plot_utils

import context
from tensorno.utils import get_all_activations


def plot_channelout_architecture(network: tf.keras.Model,
                                 inputs,
                                 ax=None,
                                 filename=None,
                                 **plot_kwargs):
    """Plots a channel out network on specified ax, with pathways indicating
    the active nodes for every datapoint in the input.

    Args:
        network (tf.keras.Model): A tf neural network model with
                                  ChannelOut hidden layers.
        input (list): Iterable of the feature inputs that generate the
                      layer activations.
        ax (ax, optional): plt ax on which to plot. Defaults to None.
    """
    if ax is None:
        _, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])

    npt.plot_nodes(network.layers, ax=ax)
    for input in inputs:
        all_activations = get_all_activations(network,
                                              input.reshape(1, -1))
        isactive = [(activations != 0.).reshape([activations.shape[-1]])
                    for activations in all_activations]
        isactive[-1] = np.where(
            all_activations[-1] == np.max(all_activations[-1]),
            True, False
        ).reshape(all_activations[-1].shape[-1])
        npt.plot_pathways(network.layers, isactive, ax=ax, **plot_kwargs)

    if filename is not None:
        plt.savefig(plot_utils.make_figs_path(filename))

    return ax


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import numpy as np

    from sknotlearn.datasets import load_MNIST, load_CIFAR10
    from tensorno.bob import build_LWTA_classifier
    from tensorno.utils import count_parameters

    x_train, y_train, x_test, y_test = load_MNIST()
    # x_train, y_train, x_test, y_test = load_CIFAR10()

    # Flattening image arrays.
    x_train = x_train.reshape((x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((x_test.shape[0], np.prod(x_test.shape[1:])))

    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_val, y_val = x_test, y_test

    """Standard model for visualisation"""
    model = build_LWTA_classifier(
        num_layers=3,
        units=[8, 8, 8],
        num_groups=[4, 2, 4],
        num_features=x_train.shape[-1],
        num_categories=y_train.shape[-1],
        dropout_rate=1/8,
        lmbda=1e-2,
        Layer="channel_out",
    )

    test_digits = np.argwhere(y_test == 1)[:, 1]
    idcs_zeros = np.argwhere(test_digits == 0)[:50]
    idcs_ones = np.argwhere(test_digits == 1)[:50]
    idcs_fours = np.argwhere(test_digits == 4)[:50]
    idcs_fives = np.argwhere(test_digits == 5)[:50]
    idcs_sevens = np.argwhere(test_digits == 7)[:50]
    idcs_eigths = np.argwhere(test_digits == 8)[:50]

    ax = plot_channelout_architecture(model, inputs=x_test[idcs_zeros],
                                      color=plot_utils.colors[1])
    plot_channelout_architecture(model, inputs=x_test[idcs_ones],
                                 ax=ax, color=plot_utils.colors[2])
    plt.savefig(plot_utils.make_figs_path("LWTA_architecture_untrained01"))

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=5,
                                                     verbose=1,
                                                     restore_best_weights=True)

    model.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[early_stopper]
    )

    model.evaluate(x_test, y_test)
    print(f"Number of parameters in network: {count_parameters(model)}")

    ax = plot_channelout_architecture(model, inputs=x_test[idcs_zeros],
                                      color=plot_utils.colors[1])
    plot_channelout_architecture(model, inputs=x_test[idcs_ones],
                                 ax=ax, color=plot_utils.colors[2])
    plt.savefig(plot_utils.make_figs_path("LWTA_architecture_trained01"))

    ax = plot_channelout_architecture(model, inputs=x_test[idcs_fours],
                                      color=plot_utils.colors[1])
    plot_channelout_architecture(model, inputs=x_test[idcs_fives],
                                 ax=ax, color=plot_utils.colors[2])
    plt.savefig(plot_utils.make_figs_path("LWTA_architecture_trained45"))

    ax = plot_channelout_architecture(model, inputs=x_test[idcs_sevens],
                                      color=plot_utils.colors[1])
    plot_channelout_architecture(model, inputs=x_test[idcs_eigths],
                                 ax=ax, color=plot_utils.colors[2])
    plt.savefig(plot_utils.make_figs_path("LWTA_architecture_trained78"))

    plt.show()
