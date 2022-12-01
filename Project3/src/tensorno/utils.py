
import tensorflow as tf
from math import sqrt


def initializers(num_inputs, bias=0.):
    result = dict()
    result["kernel_initializer"] = tf.keras.initializers.RandomNormal(
        stddev=1/sqrt(num_inputs)
    )
    result["bias_initializer"] = tf.keras.initializers.Constant(bias)
    return result
