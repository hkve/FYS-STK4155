import tensorflow as tf
import keras_tuner

if __name__ == "__main__":
    from bob import get_LWTA_architecture_builder
else:
    from tensorno.bob import get_LWTA_architecture_builder


def tune_LWTA_architecture(layer_type: str,
                           train_data: tuple,
                           val_data: tuple,
                           isregression: bool = True,
                           layer_choices: list = [2, 3, 4, 5],
                           node_choices: list = [4, 8, 16, 32],
                           group_choices: list = [1, 2, 4, 8],
                           builder_kwargs: dict = {},
                           tuner_kwargs: dict = {},
                           search_kwargs: dict = {}) -> keras_tuner.Tuner:
    """_summary_

    Args:
        layer_type (str): Specifies whether to use MaxOut or ChannelOut layers.
                          Either "max_out" or "channel_out".
        train_data (tuple): The data used for training the models on.
                            Like (inputs, targets).
        val_data (tuple): The data used for validating the models.
                          Like (inputs, targets).
        isregression (bool, optional): Whether the model is used for regression
                                       or classification. Defaults to True.
        layer_choices (list, optional): The choices for number of layers in
                                        the networks. Defaults to [2, 3, 4, 5].
        node_choices (list, optional): The choices for number of nodes in the
                                       layers of the networks.
                                       Defaults to [4, 8, 16, 32].
        group_choices (list, optional): The choices for number of groups in
                                        the layers of the networks.
                                        Defaults to [1, 2, 4, 8].
        builder_kwargs (dict, optional): kwargs to pass on to
                                         get_LWTA_architecture_builder.
        tuner_kwargs (dict, optional): kwargs to pass on to keras_tuner.
        search_kwargs (dict, optional): kwargs to pass on to
                                        keras_tuner.search.

    Returns:
        keras_tuner.Tuner: keras_tuner.Tuner instance
                           that search is completed.
    """
    # Default arguments to pass to tuner.
    _tuner_kwargs = dict(
        max_epochs=150,
        hyperband_iterations=1,
        objective="val_loss",
        directory=layer_type,
    )
    _tuner_kwargs.update(tuner_kwargs)

    tuner = keras_tuner.Hyperband(
        get_LWTA_architecture_builder(layer_type,
                                      isregressor=isregression,
                                      num_features=train_data[0].shape[-1],
                                      num_categories=train_data[1].shape[-1],
                                      layer_choices=layer_choices,
                                      node_choices=node_choices,
                                      group_choices=group_choices,
                                      **builder_kwargs),
        **_tuner_kwargs
    )

    # Default arguments to pass to search.
    _search_kwargs = dict(
        epochs=150,
    )
    _search_kwargs.update(search_kwargs)

    tuner.search(
        x=train_data[0],
        y=train_data[1],
        validation_data=val_data,
        **_search_kwargs
    )

    return tuner
