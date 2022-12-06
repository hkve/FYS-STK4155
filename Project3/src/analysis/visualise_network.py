"""Contains the code used to plot visualise LWTA NN architecture
and activation."""
import matplotlib.pyplot as plt
import tensorflow as tf

import network_plot_tools as npt
import plot_utils


def plot_channelout_architecture(network: tf.keras.Model,
                                 inputs,
                                 ax=None,
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
        all_activations = npt.get_all_activations(network,
                                                  input.reshape(1, -1))
        isactive = [(activations != 0.).reshape([activations.shape[-1]])
                    for activations in all_activations]
        isactive[-1] = np.where(
            all_activations[-1] == np.max(all_activations[-1]),
            True, False
        ).reshape(all_activations[-1].shape[-1])
        npt.plot_pathways(network.layers, isactive, ax=ax, **plot_kwargs)

    return ax


if __name__ == "__main__":
    from sknotlearn.datasets import load_Terrain, load_BreastCancer
    from sknotlearn.data import Data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_digits
    from tensorno.bob import build_LWTA_regressor, build_LWTA_classifier
    from tensorno.tuner import tune_LWTA_architecture
    import numpy as np

    # D = load_Terrain(random_state=123, n=600)
    # D = load_BreastCancer()
    x, y = load_digits(return_X_y=True, as_frame=False)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = np.array([np.where(y_ == labels, 1, 0) for y_ in y])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.75, random_state=42
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # y_train = np.c_[y_train, 1-y_train]
    # y_test = np.c_[y_test, 1-y_test]

    # tuner = tune_LWTA_architecture(
    #     layer_type="channel_out",
    #     train_data=(x_train, y_train),
    #     val_data=(x_test, y_test),
    #     isregression=False,
    #     layer_choices=[2, 3, 4, 5],
    #     node_choices=[4, 8, 16, 32],
    #     group_choices=[1, 2, 4, 8],
    #     _tuner_kwargs=dict(
    #         max_epochs=50,
    #         hyperband_iterations=3,
    #         project_name="test1",
    #         # overwrite=True
    #     ),
    #     _search_kwargs=dict(
    #         epochs=50
    #     )
    # )

    # tuner.results_summary(num_trials=3)

    """Standard model for visualisation"""
    # model = build_LWTA_classifier(
    #     num_layers=3,
    #     units=[8, 8, 8],
    #     num_groups=[2, 4, 4],
    #     num_features=x_train.shape[-1],
    #     num_categories=len(labels),
    #     Layer="channel_out",
    # )

    model = build_LWTA_classifier(
        num_layers=3,
        units=[32, 16, 8],
        num_groups=[4, 8, 8],
        num_features=x_train.shape[-1],
        num_categories=len(labels),
        Layer="channel_out",
    )
    # model = build_LWTA_classifier(
    #     num_layers=2,
    #     units=[2**3, 2**4],
    #     num_groups=[2**3, 2**3],
    #     num_features=x_train.shape[-1],
    #     num_categories=len(labels),
    #     Layer="channel_out",
    # )

    test_digits = np.argwhere(y_test == 1)[:, 1]
    idcs_ones = np.argwhere(test_digits == 1)
    idcs_fours = np.argwhere(test_digits == 4)

    # ax = plot_channelout_architecture(model, inputs=x_test[idcs_ones],
    #                                   color=plot_utils.colors[1])
    # plot_channelout_architecture(model, inputs=x_test[idcs_fours], ax=ax,
    #                              color=plot_utils.colors[2])
    # plt.show()

    tf.random.set_seed(321)
    model.fit(
        x_train, y_train,
        epochs=50,  # 500,
        validation_data=(x_test, y_test)
    )

    ax = plot_channelout_architecture(model, inputs=x_test[idcs_ones],
                                      color=plot_utils.colors[1])
    plot_channelout_architecture(model, inputs=x_test[idcs_fours], ax=ax,
                                 color=plot_utils.colors[2])
    plt.show()
