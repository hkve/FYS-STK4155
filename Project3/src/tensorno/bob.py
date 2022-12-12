"""Contains builders used for creating maxout and channelout neural networks
in tensorflow"""
# TODO: Comment and document better.
import tensorflow as tf
import keras_tuner

if __name__ == "__main__":
    from layers import MaxOut, ChannelOut
    from utils import get_custom_initializers
else:
    from tensorno.layers import MaxOut, ChannelOut
    from tensorno.utils import get_custom_initializers


def build_FFNN_classifier(
    num_layers: int,
    units: tuple,
    activation: str,
    num_features: int = 2,
    num_categories: int = 1,
    dropout_rate: float = None,
    lmbda: float = None,
    **compile_kwargs
) -> tf.keras.Sequential:
    """Builds a LWTA FFNN for classification with the specified architecture.

    Args:
        num_layers (int): Number of hidden layers.
        units (tuple): Iterable with the number of nodes for each hidden layer.
        activation (str): The name of the activation function to use.
        num_features (int, optional): Number of input features. Defaults to 2.
        num_categories (int, optional): Number of output categories.
                                        Defaults to 1.
        dropout_rate (float, optional): Rate with which to use dropout
                                        between layers. None means no dropout.
                                        Defaults to None.
        lmbda (float, optional): L2 penalisation parameters on the kernel
                                 parameters of the layers. None means
                                 no penalisation. Defaults to None.

    Returns:
        tf.keras.Sequential: Compiled Sequential model as classifier.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(num_features,),
        name="input"
    ))
    if dropout_rate is not None:
        model.add(
            tf.keras.layers.Dropout(
                dropout_rate,
                input_shape=(num_features,)
            )
        )
    if lmbda is not None:
        weight_penalizer = tf.keras.regularizers.L2(lmbda)
    else:
        weight_penalizer = None

    # Add hidden layers.
    num_inputs = num_features
    for layer in range(num_layers):
        model.add(tf.keras.layers.Dense(
            units=units[layer],
            input_shape=(num_inputs,),
            activation=activation,
            **get_custom_initializers(num_inputs),
            kernel_regularizer=weight_penalizer,
            name=f"{activation}_{layer+1}"
        ))

        num_inputs = units[layer]
        if dropout_rate is not None:
            model.add(
                tf.keras.layers.Dropout(
                    dropout_rate,
                    input_shape=(num_inputs,)
                )
            )

    # Add output layer.
    model.add(tf.keras.layers.Dense(
        units=num_categories,
        activation="sigmoid" if num_categories == 1 else "softmax",
        **get_custom_initializers(num_inputs),
        name="output"
    ))

    kwargs = dict(
        optimizer="adam",
        loss=("binary" if num_categories == 1 else "categorical")
        + "_crossentropy",
        metrics=["accuracy"]
    )
    kwargs.update(compile_kwargs)
    model.compile(**kwargs)

    return model


def build_LWTA_regressor(
    num_layers: int,
    units: tuple,
    num_groups: tuple,
    Layer: str,
    num_features: int = 2,
    dropout_rate: float = None,
    lmbda: float = None,
    **compile_kwargs
) -> tf.keras.Sequential:
    """Builds a LWTA FFNN for regression with the specified architecture.

    Args:
        num_layers (int): Number of hidden layers.
        units (tuple): Iterable with the number of nodes for each hidden layer.
        num_groups (tuple): Iterable with the number of competing groups
                            for each hidden layer.
        Layer (str): Specifies whether to use MaxOut or ChannelOut layers.
                     Either "max_out" or "channel_out".
        num_features (int, optional): Number of input features. Defaults to 2.
        dropout_rate (float, optional): Rate with which to use dropout
                                        between layers. None means no dropout.
                                        Defaults to None.
        lmbda (float, optional): L2 penalisation parameters on the kernel
                                 parameters of the layers. None means
                                 no penalisation. Defaults to None.

    Returns:
        tf.keras.Sequential: Compiled Sequential model as regressor.
    """
    # Get the appropriate Layer class
    if Layer == "max_out":
        Layer = MaxOut
    elif Layer == "channel_out":
        Layer = ChannelOut
    else:
        raise ValueError(f"Given Layer ({Layer}) is not supported. "
                         "must be `max_out` or `channel_out`.")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(num_features,),
        name="input"
    ))
    if dropout_rate is not None:
        model.add(
            tf.keras.layers.Dropout(
                dropout_rate,
                input_shape=(num_features,)
            )
        )
    if lmbda is not None:
        weight_penalizer = tf.keras.regularizers.L2(lmbda)
    else:
        weight_penalizer = None

    # Add hidden layers.
    num_inputs = num_features
    for layer in range(num_layers):
        if num_groups[layer] > units[layer]:
            num_groups[layer] = units[layer]
        if num_groups[layer] == units[layer]:
            model.add(tf.keras.layers.Dense(
                units=units[layer],
                input_shape=(num_inputs,),
                activation="ReLU",
                **get_custom_initializers(
                    num_features if layer == 1 else num_groups[layer-1]
                ),
                kernel_regularizer=weight_penalizer,
                name=f"ReLU_{layer+1}"
            ))
        else:
            model.add(Layer(
                units=units[layer],
                num_inputs=num_inputs,
                num_groups=num_groups[layer],
                **get_custom_initializers(
                    num_features if layer == 1 else num_groups[layer-1]
                ),
                kernel_regularizer=weight_penalizer,
                name=f"{Layer.__name__.lower()}_{layer+1}"
            ))

        # MaxOut and ChannelOut have different number of outputs.
        if isinstance(Layer, MaxOut):
            num_inputs = num_groups[layer]
        else:
            num_inputs = units[layer]

        if dropout_rate is not None:
            model.add(
                tf.keras.layers.Dropout(
                    dropout_rate,
                    input_shape=(num_inputs,)
                )
            )

    # Add output layer.
    model.add(tf.keras.layers.Dense(
        units=1,
        activation="linear",
        **get_custom_initializers(num_inputs),
        name="output"
    ))

    kwargs = dict(
        optimizer="adam",
        loss="mse"
    )
    kwargs.update(compile_kwargs)
    model.compile(**kwargs)

    return model


def build_LWTA_classifier(
    num_layers: int,
    units: tuple,
    num_groups: tuple,
    Layer: str,
    num_features: int = 2,
    num_categories: int = 1,
    dropout_rate: float = None,
    lmbda: float = None,
    **compile_kwargs
) -> tf.keras.Sequential:
    """Builds a LWTA FFNN for classification with the specified architecture.

    Args:
        num_layers (int): Number of hidden layers.
        units (tuple): Iterable with the number of nodes for each hidden layer.
        num_groups (tuple): Iterable with the number of competing groups
                            for each hidden layer.
        Layer (str): Specifies whether to use MaxOut or ChannelOut layers.
                     Either "max_out" or "channel_out".
        num_features (int, optional): Number of input features. Defaults to 2.
        num_categories (int, optional): Number of output categories.
                                        Defaults to 2.
        dropout_rate (float, optional): Rate with which to use dropout
                                        between layers. None means no dropout.
                                        Defaults to None.
        lmbda (float, optional): L2 penalisation parameters on the kernel
                                 parameters of the layers. None means
                                 no penalisation. Defaults to None.

    Returns:
        tf.keras.Sequential: Compiled Sequential model as classifier.
    """
    # Get the appropriate Layer class
    if Layer == "max_out":
        Layer = MaxOut
    elif Layer == "channel_out":
        Layer = ChannelOut
    else:
        raise ValueError(f"Given Layer ({Layer}) is not supported. "
                         "must be `max_out` or `channel_out`.")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(
        input_shape=(num_features,),
        name="input"
    ))
    if dropout_rate is not None:
        model.add(
            tf.keras.layers.Dropout(
                dropout_rate,
                input_shape=(num_features,)
            )
        )
    if lmbda is not None:
        weight_penalizer = tf.keras.regularizers.L2(lmbda)
    else:
        weight_penalizer = None

    # Add hidden layers.
    num_inputs = num_features
    for layer in range(num_layers):
        if num_groups[layer] > units[layer]:
            num_groups[layer] = units[layer]
        if num_groups[layer] == units[layer]:
            model.add(tf.keras.layers.Dense(
                units=units[layer],
                input_shape=(num_inputs,),
                activation="ReLU",
                **get_custom_initializers(
                    num_features if layer == 1 else num_groups[layer-1]
                ),
                kernel_regularizer=weight_penalizer,
                name=f"ReLU_{layer+1}"
            ))
        else:
            model.add(Layer(
                units=units[layer],
                num_inputs=num_inputs,
                num_groups=num_groups[layer],
                **get_custom_initializers(
                    num_features if layer == 1 else num_groups[layer-1]
                ),
                kernel_regularizer=weight_penalizer,
                name=f"{Layer.__name__.lower()}_{layer+1}"
            ))

        # MaxOut and ChannelOut have different number of outputs.
        if Layer is MaxOut:
            num_inputs = num_groups[layer]
        else:
            num_inputs = units[layer]

        if dropout_rate is not None:
            model.add(
                tf.keras.layers.Dropout(
                    dropout_rate,
                    input_shape=(num_inputs,)
                )
            )

    # Add output layer.
    model.add(tf.keras.layers.Dense(
        units=num_categories,
        activation="sigmoid" if num_categories == 1 else "softmax",
        **get_custom_initializers(num_inputs),
        name="output"
    ))

    kwargs = dict(
        optimizer="adam",
        loss=("binary" if num_categories == 1 else "categorical")
        + "_crossentropy",
        metrics=["accuracy"]
    )
    kwargs.update(compile_kwargs)
    model.compile(**kwargs)

    return model


def build_LWTA_architecture(
    hp: keras_tuner.HyperParameters,
    Layer: str,
    layer_choices: list = [2, 3, 4, 5],
    node_choices: list = [4, 8, 16, 32],
    group_choices: list = [1, 2, 4, 8],
    isregressor: bool = True,
    num_features: int = None,
    num_categories: int = None,
    dropout_rate: float = None,
    lmbda: float = None
) -> tf.keras.Sequential:
    """Builder for interfacing with keras_tuner to tune model architecture.

    Args:
        hp (keras_tuner.HyperParameters): Stores and generates hyperparameters
                                          for the model.
        Layer (str): Specifies whether to use MaxOut or ChannelOut layers.
                     Either "max_out" or "channel_out".
        layer_choices (list, optional): The choices for number of layers in
                                        the networks. Defaults to [2, 3, 4, 5].
        node_choices (list, optional): The choices for number of nodes in the
                                       layers of the networks.
                                       Defaults to [4, 8, 16, 32].
        group_choices (list, optional): The choices for number of groups in
                                        the layers of the networks.
                                        Defaults to [1, 2, 4, 8].
        isregressor (bool, optional): Whether the model is used for regression
                                      or classification. Defaults to True.
        num_features (int, optional): Number of input features.
                                      Defaults to None.
        num_categories (int, optional): Number of output categories.
                                        Defaults to None.
        dropout_rate (float, optional): Rate with which to use dropout
                                        between layers. None means no dropout.
                                        Defaults to None.
        lmbda (float, optional): L2 penalisation parameters on the kernel
                                 parameters of the layers. None means
                                 no penalisation. Defaults to None.

    Returns:
        tf.keras.Sequential: Sequential model with a choice for architecture.
    """
    num_layers = hp.Choice("num_layers", layer_choices)
    units = list()
    num_groups = list()
    for layer in range(num_layers):
        with hp.conditional_scope(
            "num_layers", list(range(layer + 1, max(layer_choices) + 1))
        ):
            units.append(hp.Choice(f"num_nodes_{layer+1}", node_choices))
            num_groups.append(hp.Choice(f"num_groups_{layer+1}", group_choices))

    if isregressor:
        return build_LWTA_regressor(num_layers, units, num_groups, Layer,
                                    num_features, dropout_rate, lmbda)
    else:
        return build_LWTA_classifier(num_layers, units, num_groups, Layer,
                                     num_features, num_categories,
                                     dropout_rate, lmbda)


def get_LWTA_architecture_builder(
    Layer: str,
    layer_choices: list = [2, 3, 4, 5],
    node_choices: list = [4, 8, 16, 32],
    group_choices: list = [1, 2, 4, 8],
    isregressor: bool = True,
    num_features: int = None,
    num_categories: int = None,
    dropout_rate: float = None,
    lmbda: float = None
):
    """Returns appropriate architecture builder for interfacing with
    keras_tuner for tuning the architecture of a LWTA model.

    Args:
        Layer (str): Specifies whether to use MaxOut or ChannelOut layers.
                     Either "max_out" or "channel_out".
        layer_choices (list, optional): The choices for number of layers in
                                        the networks. Defaults to [2, 3, 4, 5].
        node_choices (list, optional): The choices for number of nodes in the
                                       layers of the networks.
                                       Defaults to [4, 8, 16, 32].
        group_choices (list, optional): The choices for number of groups in
                                        the layers of the networks.
                                        Defaults to [1, 2, 4, 8].
        isregressor (bool, optional): Whether the model is used for regression
                                      or classification. Defaults to True.
        num_features (int, optional): Number of input features.
                                      Defaults to None.
        num_categories (int, optional): Number of output categories.
                                        Defaults to None.
        dropout_rate (float, optional): Rate with which to use dropout
                                        between layers. None means no dropout.
                                        Defaults to None.
        lmbda (float, optional): L2 penalisation parameters on the kernel
                                 parameters of the layers. None means
                                 no penalisation. Defaults to None.

    Returns:
        function: Builder that keras_tuner can use for hyperparameter search.
    """
    return lambda hp: build_LWTA_architecture(hp,
                                              Layer=Layer,
                                              layer_choices=layer_choices,
                                              node_choices=node_choices,
                                              group_choices=group_choices,
                                              isregressor=isregressor,
                                              num_features=num_features,
                                              num_categories=num_categories,
                                              dropout_rate=dropout_rate,
                                              lmbda=lmbda)
