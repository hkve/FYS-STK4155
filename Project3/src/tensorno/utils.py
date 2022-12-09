"""Contains various utility functions used in the tensorno library."""

import tensorflow as tf
import numpy as np
from math import sqrt


def get_custom_initializers(num_inputs: int, bias: float = 0.) -> dict:
    """Returns a kwargs dict that can be passed to tf.keras.Dense-type layers
    with kernel and bias initializers. 'He' initialization is implemented where
    the kernel is initialized with a normal distribution weighted with the
    number of inputs. The bias is initialized to the same constant value.

    Args:
        num_inputs (int): Number of inputs to the layer.
        bias (float, optional): Bias terms' initial value. Defaults to 0..

    Returns:
        dict: kwargs to be passed to contructor of tf.keras.layers.Dense.
    """
    result = dict(
        kernel_initializer=tf.keras.initializers.RandomNormal(
            stddev=sqrt(2./num_inputs)
        ),
        bias_initializer=tf.keras.initializers.Constant(bias)
    )

    return result


def count_parameters(network: tf.keras.Model):
    counter = 0
    for layer in network.layers:
        for variable in layer._trainable_weights:
            counter += np.prod(variable.get_shape())
    return counter


def get_layer_names(network: tf.keras.Model) -> list:
    """Get the string names of the layers of a tf Model instance.

    Args:
        network (tf.keras.Model): Model from which to get layer names.

    Returns:
        list: list of the string names of the layers in network.
    """
    return list(map(lambda x: x.name, network.layers))


def get_layer_activations(network: tf.keras.Model,
                          layer,
                          input):
    """Get the output from a given layer in a tf Model given input.

    Args:
        network (tf.keras.Model): Model from which to extract activation.
        layer (str or tf.keras.layers.Layer): Either the name of the layer or
                                              the Layer instance from which to
                                              get the outputs.
        input (np.ndarray or tf.Tensor): The initial feature input to generate
                                         the activations.

    Returns:
        np.ndarray or tf.Tensor: Output of the given layer.
    """
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


def get_all_activations(network: tf.keras.Model, input) -> list:
    """Gets the output from every layer in a tf Model instance given input.

    Args:
        network (tf.keras.Model): Model from which to extract activation.
        input (np.ndarray or tf.Tensor): The initial feature input to generate
                                         the activations.

    Returns:
        list: list of np.ndarrays or tf.Tensors with the output from
              every layer.
    """
    activations = list()
    for layer in network.layers:
        activations.append(get_layer_activations(network, layer, input))

    return activations
