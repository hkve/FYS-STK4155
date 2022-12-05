import tensorflow as tf
import keras_tuner

if __name__ == "__main__":
    from bob import LWTA_architecture_builder
else:
    from tensorno.bob import LWTA_architecture_builder


def tune_LWTA_architecture(layer_type: str,
                           train_data: tuple,
                           val_data: tuple,
                           isregression: bool = True,
                           layer_choices: list = [2, 3, 4, 5],
                           node_choices: list = [4, 8, 16, 32],
                           group_choices: list = [1, 2, 4, 8],
                           _tuner_kwargs: dict = None,
                           _search_kwargs: dict = None) -> keras_tuner.Tuner:
    tuner_kwargs = dict(
        max_epochs=150,
        hyperband_iterations=1,
        objective="val_loss",
        directory=layer_type,
    )
    if _tuner_kwargs is not None:
        tuner_kwargs.update(_tuner_kwargs)

    tuner = keras_tuner.Hyperband(
        LWTA_architecture_builder(layer_type,
                                  isregressor=isregression,
                                  num_features=train_data[0].shape[-1],
                                  num_categories=train_data[1].shape[-1],
                                  layer_choices=layer_choices,
                                  node_choices=node_choices,
                                  group_choices=group_choices),
        **tuner_kwargs
    )
    search_kwargs = dict(
        epochs=150,
    )
    if _search_kwargs is not None:
        search_kwargs.update(_search_kwargs)

    tuner.search(
        x=train_data[0],
        y=train_data[1],
        validation_data=val_data,
        **search_kwargs
    )

    return tuner
