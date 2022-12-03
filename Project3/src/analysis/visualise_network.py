import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import plot_utils
import context
import tensorno as tn
from tensorno.layers import MaxOut, ChannelOut
from tensorno.bob import build_LWTA_regressor


def get_layer_names(network):
    return list(map(lambda x: x.name, network.layers))


def get_layer_activations(network, layer, input):
    if isinstance(layer, str):
        intermediate_model = tf.keras.Model(network.input,
                                            network.get_layer(layer).output,
                                            name="intermediate")
    elif isinstance(layer, tf.keras.layers.Layer):
        intermediate_model = tf.keras.Model(network.input,
                                            layer.output,
                                            name="intermediate")
    else:
        raise ValueError("Given layer is not a string or Layer instance"
                         f" (is type {type(layer)})")
    if isinstance(input, np.ndarray):
        return intermediate_model(input).numpy()
    else:
        return intermediate_model(input)


def get_all_activations(network, input):
    activations = list()
    for layer in network.layers:
        activations.append(get_layer_activations(network, layer, input))
    return activations


def plot_channelout_architecture(network, input):
    fig, ax = plt.subplots()
    for x, layer in enumerate(model.layers):
        nodes = layer.units
        activations = get_layer_activations(network, layer, input)
        isactive = (activations != 0.).reshape([activations.shape[-1]])
        colors = np.where(isactive, "r", "b")
        for node, color in zip(np.arange(-nodes/2, nodes/2), colors):
            ax.scatter([x], [node], c=color)

        if isinstance(layer, ChannelOut):
            groups = layer.num_groups
            competitors = nodes // groups
            for group in range(groups):
                ax.add_patch(plt.Rectangle((x-0.1,
                                            group*competitors - nodes/2 - 0.3),
                                           0.2, competitors-0.4,
                                           fc='none',
                                           ec='grey',
                                           lw=5,
                                           clip_on=False))


if __name__ == "__main__":
    from sknotlearn.datasets import load_Terrain
    D = load_Terrain(random_state=123, n=600)
    D_train, D_test = D.train_test_split(ratio=0.75, random_state=42)
    D_train = D_train.scaled(scheme="Standard")
    D_test = D_train.scale(D_test)
    y_train, x_train = D_train.unpacked()
    y_test, x_test = D_test.unpacked()

    model = build_LWTA_regressor(
        num_layers=2,
        units=[8, 8],
        num_groups=[2, 4],
        # activation=tn.activations.MaxOut
        Layer="channel_out"
    )

    # model.summary()
    # exit()

    # for activations in get_all_activations(model, x_test[0:1]):
    #     print(activations)
    # exit()

    plot_channelout_architecture(model, input=x_test[0:1])
    plt.show()

    # early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
    #                                                  patience=50,
    #                                                  verbose=1,
    #                                                  restore_best_weights=True)
    # tf.random.set_seed(321)
    # model.fit(
    #     x=x_train,
    #     y=y_train,
    #     epochs=500,
    #     validation_data=(x_test, y_test),
    #     callbacks=[early_stopper],
    #     verbose=0
    # )
    # model.evaluate(x_test, y_test)

    # for activations in get_all_activations(model, x_test[:1]):
    #     print(activations)
