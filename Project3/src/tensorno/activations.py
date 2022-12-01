import tensorflow as tf


class Activation:
    def __init__(self):
        pass

    @tf.function
    def __call__(self, inputs):
        raise NotImplementedError("Activation class is not callable")


class Max_out(Activation):
    def __init__(self, num_groups=2):
        super().__init__()
        self.num_groups = num_groups

    @tf.function
    def __call__(self, inputs):
        num_inputs, num_units = inputs.shape
        if num_inputs is None:
            num_inputs = -1
        if self.num_groups > num_units:
            self.num_groups = num_units
        elif num_units % self.num_groups != 0:
            raise ValueError(f"Number of units ({num_units}) "
                             "is not divisible by number of groups "
                             f"({self.num_groups})")
        num_competitors = num_units // self.num_groups
        new_shape = [num_inputs, self.num_groups, num_competitors]
        inputs = tf.reshape(inputs, new_shape)
        outputs = tf.math.reduce_max(inputs, axis=-1)
        # outputs = tf.reshape(outputs, new_shape[:-1])
        return outputs


class Channel_out(Activation):
    def __init__(self, num_groups=2):
        super().__init__()
        self.num_groups = num_groups

    @tf.function
    def __call__(self, inputs):
        num_inputs, num_units = inputs.shape
        if num_inputs is None:
            num_inputs = -1
        if self.num_groups > num_units:
            self.num_groups = num_units
        elif num_units % self.num_groups != 0:
            raise ValueError(f"Number of units ({num_units}) "
                             "is not divisible by number of groups "
                             f"({self.num_groups})")
        num_competitors = num_units // self.num_groups
        new_shape = [num_inputs, self.num_groups, num_competitors]
        inputs = tf.reshape(inputs, new_shape)
        outputs = tf.math.reduce_max(inputs, axis=-1, keepdims=True)
        outputs = tf.where(tf.equal(inputs, outputs), inputs, 0)
        outputs = tf.reshape(outputs, [num_inputs, num_units])
        return outputs


@tf.function
def max_out(inputs, num_groups=2):
    num_inputs, num_units = inputs.shape
    if num_inputs is None:
        num_inputs = -1
    if num_groups > num_units:
        num_groups = num_units
    elif num_units % num_groups != 0:
        raise ValueError(f"Number of units ({num_units}) "
                         f"is not divisible by number of groups ({num_groups})")
    num_competitors = num_units // num_groups
    new_shape = [num_inputs, num_groups, num_competitors]
    inputs = tf.reshape(inputs, new_shape)
    outputs = tf.math.reduce_max(inputs, axis=-1)
    # outputs = tf.reshape(outputs, new_shape[:-1])
    return outputs


@tf.function
def channel_out(inputs, num_groups=2):
    num_inputs, num_units = inputs.shape
    if num_inputs is None:
        num_inputs = -1
    if num_groups > num_units:
        num_groups = num_units
    elif num_units % num_groups != 0:
        raise ValueError(f"Number of units ({num_units}) "
                         f"is not divisible by number of groups ({num_groups})")
    num_competitors = num_units // num_groups
    new_shape = [num_inputs, num_groups, num_competitors]
    inputs = tf.reshape(inputs, new_shape)
    outputs = tf.math.reduce_max(inputs, axis=-1, keepdims=True)
    outputs = tf.where(tf.equal(inputs, outputs), inputs, 0)
    outputs = tf.reshape(outputs, [num_inputs, num_units])
    return outputs


if __name__ == "__main__":
    pass
