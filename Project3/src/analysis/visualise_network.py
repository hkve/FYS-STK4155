import tensorflow as tf
import matplotlib.pyplot as plt

import plot_utils
import context
from tensorno.activations import Max_out, Channel_out
from tensorno.bob import build_from_architecture


def get_activations(network, layer_name, input):
    intermediate_model = tf.keras.Model(network.input,
                                        network.get_layer(layer_name).output,
                                        name="intermediate")
    return intermediate_model(input)


if __name__ == "__main__":
    from sknotlearn.datasets import load_Terrain
    D = load_Terrain(random_state=123, n=600)
    D_train, D_test = D.train_test_split(ratio=0.75, random_state=42)
    D_train = D_train.scaled(scheme="Standard")
    D_test = D_train.scale(D_test)
    y_train, x_train = D_train.unpacked()
    y_test, x_test = D_test.unpacked()

    model = build_from_architecture(
        num_layers=2,
        units=[8, 8],
        num_groups=[2, 4],
        activation=Channel_out
    )

    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                     patience=50,
                                                     verbose=1,
                                                     restore_best_weights=True)
    tf.random.set_seed(321)
    model.fit(
        x=x_train,
        y=y_train,
        epochs=500,
        validation_data=(x_test, y_test),
        callbacks=[early_stopper],
        verbose=0
    )
    model.evaluate(x_test, y_test)

    '''
    layers = list(map(lambda x: x.name, model.layers))
    print(layers)
    input = x_test[0:1]
    # output = get_activations(model, layers[0], input).numpy()

    # weights2 = model.get_layer(layers[1]).get_weights()[0]
    # input = output @ weights2
    # print(input)

    for layer in layers:
        output = get_activations(model, layer, input).numpy()
        print(layer, output)
    '''
