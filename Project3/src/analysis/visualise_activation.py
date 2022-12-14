import numpy as np
import matplotlib.pyplot as plt

import plot_utils


def plot_approximation(func, dfunc, x, num_nodes, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_xticks([])
    ax.set_yticks([])

    x_nodes = np.linspace(min(x), max(x), num_nodes+2)[1:-1]

    ax.plot(x, func(x), c=plot_utils.colors[5])
    for x_node in x_nodes:
        def node_func(x): return func(x_node) + dfunc(x_node) * (x - x_node)
        ax.plot(x, node_func(x), ls="--")

    return ax


if __name__ == "__main__":
    def ReLU(x):
        return np.where(x > 0, x, 0)

    def dReLU(x):
        return np.where(x > 0, 1, 0)

    def dabs(x):
        return np.where(x > 0, 1, -1)

    _, axes = plt.subplots(1, 3)

    plot_approximation(
        func=ReLU,
        dfunc=dReLU,
        x=np.linspace(-1, 1, 101),
        num_nodes=2,
        ax=axes[0]
    )
    plot_approximation(
        func=abs,
        dfunc=dabs,
        x=np.linspace(-1, 1, 101),
        num_nodes=2,
        ax=axes[1]
    )
    plot_approximation(
        func=lambda x: x**2,
        dfunc=lambda x: 2*x,
        x=np.linspace(-1, 1, 101),
        num_nodes=3,
        ax=axes[2]
    )
    plt.savefig(plot_utils.make_figs_path("maxout_activations"))
    plt.show()
