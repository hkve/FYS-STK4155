"""Contains the MaxOut and ChannelOut Layer classes
interfaced with tensorflow"""

import tensorflow as tf

# from typing import Optional
from tensorflow.keras.layers import Layer
from tensorflow.python.ops import gen_math_ops, nn_ops


class MaxOut(Layer):
    """``MaxOut``."""

    def __init__(
        self,
        units: int,
        num_inputs: int,
        num_groups: int = 2,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        **kwargs
    ):
        # Using Layer initialization
        super().__init__(**kwargs)

        # Initializing units and groups
        if num_groups > units:
            num_groups = units
        elif units % num_groups != 0:
            raise ValueError(f"Number of units ({units}) "
                             "is not divisible by number of groups "
                             f"({num_groups})")
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.num_groups = int(num_groups) if not isinstance(num_groups, int) \
            else num_groups
        if self.num_groups < 0:
            raise ValueError(
                "Received an invalid value for `num_groups`, expected "
                f"a positive integer. Received: num_groups={num_groups}"
            )

        # Initializing weights and biases
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.kernel = self.add_weight(
            "kernel",
            shape=[num_inputs, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=tf.float32,
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.units, ],
            initializer=self.bias_initializer,
            dtype=tf.float32,
            trainable=True,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Passing input through weight kernel and adding bias terms
        inputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
        inputs = nn_ops.bias_add(inputs, self.bias)

        num_inputs = inputs.shape[0]
        if num_inputs is None:
            num_inputs = -1

        # Reshaping outputs such that they are grouped correctly
        num_competitors = self.units // self.num_groups
        new_shape = [num_inputs, self.num_groups, num_competitors]
        inputs = tf.reshape(inputs, new_shape)
        # Finding maximum activation in each group
        outputs = tf.math.reduce_max(inputs, axis=-1)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_groups": self.num_groups,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                )
            }
        )
        return config


class ChannelOut(Layer):
    """``ChannelOut``."""

    def __init__(
        self,
        units: int,
        num_inputs: int,
        num_groups: int = 2,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Initializing units and groups
        if num_groups > units:
            num_groups = units
        elif units % num_groups != 0:
            raise ValueError(f"Number of units ({units}) "
                             "is not divisible by number of groups "
                             f"({num_groups})")
        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                "Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )
        self.num_groups = int(num_groups) if not isinstance(num_groups, int) \
            else num_groups
        if self.num_groups < 0:
            raise ValueError(
                "Received an invalid value for `num_groups`, expected "
                f"a positive integer. Received: num_groups={num_groups}"
            )

        # Initializing weights and biases
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)

        self.kernel = self.add_weight(
            "kernel",
            shape=[num_inputs, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            dtype=tf.float32,
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.units, ],
            initializer=self.bias_initializer,
            dtype=tf.float32,
            trainable=True,
        )

    def call(self,
             inputs: tf.Tensor,
             mask: tf.Tensor = None
             ) -> tf.Tensor:
        # Pass input through weight kernel and adding bias terms.
        inputs = gen_math_ops.MatMul(a=inputs, b=self.kernel)
        inputs = nn_ops.bias_add(inputs, self.bias)

        num_inputs = inputs.shape[0]
        if num_inputs is None:
            num_inputs = -1

        # Reshaping inputs such that they are grouped correctly
        num_competitors = self.units // self.num_groups
        new_shape = [num_inputs, self.num_groups, num_competitors]
        inputs = tf.reshape(inputs, new_shape)

        # Finding maximum activations and setting losers to 0.
        outputs = tf.math.reduce_max(inputs, axis=-1, keepdims=True)
        outputs = tf.where(tf.equal(inputs, outputs), inputs, 0.)
        # Reshaping outputs to original input shape
        outputs = tf.reshape(outputs, [num_inputs, self.units])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_groups": self.num_groups,
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self.kernel_regularizer
                )
            }
        )
        return config
