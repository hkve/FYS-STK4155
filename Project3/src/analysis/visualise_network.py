"""Contains the code used to plot visualise LWTA NN architecture
and activation."""
import matplotlib.pyplot as plt

import network_plot_tools as npt


def plot_channelout_architecture(network, inputs, **plot_kwargs):
    _, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    npt.plot_nodes(network.layers, ax=ax)
    for input in inputs:
        all_activations = npt.get_all_activations(network,
                                                  input.reshape(1, -1))
        isactive = [(activations != 0.).reshape([activations.shape[-1]])
                    for activations in all_activations]
        npt.plot_pathways(network.layers, isactive, ax=ax, **plot_kwargs)


if __name__ == "__main__":
    from sknotlearn.datasets import load_Terrain
    from tensorno.bob import build_LWTA_regressor

    D = load_Terrain(random_state=123, n=600)
    D_train, D_test = D.train_test_split(ratio=0.75, random_state=42)
    D_train = D_train.scaled(scheme="Standard")
    D_test = D_train.scale(D_test)
    y_train, x_train = D_train.unpacked()
    y_test, x_test = D_test.unpacked()

    model = build_LWTA_regressor(
        num_layers=3,
        units=[8, 8, 4],
        num_groups=[2, 4, 2],
        Layer="channel_out"
    )

    model.fit(x_train, y_train,
              epochs=500,
              validation_data=(x_test, y_test))

    plot_channelout_architecture(model, inputs=x_test)
    plt.show()
