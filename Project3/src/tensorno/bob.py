import tensorflow as tf

from tensorno.activations import Activation, Max_out, Channel_out
from tensorno.utils import initializers


def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Choice("stop", [3, 4, 5])):
        model.add(tf.keras.layers.Dense(
            hp.Choice("units", [64, 128, 256]),
            activation=Max_out(hp.Choice("num_groups", [8, 64, 128]))
        ))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [0.001, 0.01]),
        ),
        loss="mse"
    )
    return model


def build_from_architecture(num_layers, units, num_groups, activation):
    if not issubclass(activation, Activation):
        raise ValueError(
            f"Given activation ({activation}) is not an instance "
            "of the Activation class."
        )
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(2,),
        name="input"
    ))
    for layer in range(num_layers):
        if layer == 0:
            num_inputs = 2
        else:
            num_inputs = num_groups[layer-1]
        model.add(tf.keras.layers.Dense(
            units=units[layer],
            activation=activation(num_groups[layer]),
            **initializers(num_inputs),
            name=f"{activation.__name__}_{layer+1}"
                #  f"_{units[layer]}n_{num_groups[layer]}g"
        ))
    model.add(tf.keras.layers.Dense(
        units=1,
        activation="linear",
        **initializers(num_groups[-1]),
        name="output"
    ))
    model.compile(
        optimizer="adam",
        loss="mse"
    )
    return model


def build_model_architecture(hp, activation):
    num_layers = hp.Int("num_layers", 2, 4)
    units = list()
    num_groups = list()
    nodes_by_layer = [hp.Int(f"log2(num_nodes{i})", 4, 7)
                      for i in range(1, num_layers+1)]
    groups_by_layer = [hp.Int(f"log2(num_groups{i})", 3, 6)
                       for i in range(1, num_layers+1)]
    for nodes, groups in zip(nodes_by_layer, groups_by_layer):
        units.append(2**nodes)
        num_groups.append(2**groups)
    return build_from_architecture(num_layers, units, num_groups, activation)


def architecture_builder(activation):
    if activation == "max_out":
        activation = Max_out
    elif activation == "channel_out":
        activation = Channel_out
    else:
        raise ValueError(f"{activation} not a valid activation function. "
                         "Available are: 'max_out', 'channel_out'")
    return lambda hp: build_model_architecture(hp, activation)
