"""Contains useful functions for plotting neural network architecture."""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import context
import plot_utils
from tensorno.layers import MaxOut, ChannelOut


def group_nodes(x: float, nodes: int, groups: int, ax):
    """Plots rectangles around each group of competitors in a LWTA layer.

    Args:
        x (float): x-position of the layer on ax.
        nodes (int): Number of nodes in the layer.
        groups (int): Number of groups in the layer.
        ax (ax): plt ax object on which to plot.

    Returns:
        tuple: ax object on which the function plotted.
    """
    competitors = nodes // groups
    for group in range(groups):
        ax.add_patch(plt.Rectangle((x-0.1,
                                    group*competitors - nodes/2 - 0.3),
                                   0.2, competitors-0.4,
                                   fc='none',
                                   ec=plot_utils.colors[-1],
                                   lw=1.2,
                                   clip_on=False))

    return ax


def plot_nodes(layers: list, ax=None):
    """Plots all the nodes in every layer in layers, grouping LWTA groups.

    Args:
        layers (list): list of tf.keras.layers.Layer instances.
        ax (ax, optional): plt ax on which to plot. Defaults to None.

    Returns:
        tuple: ax object on which the function plotted.
    """
    if ax is None:
        _, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])

    x = 0  # Position to plot current layer
    for layer in layers:
        try:
            nodes = layer.units
        except AttributeError:
            continue

        for node in np.arange(-nodes/2, nodes/2):
            ax.scatter([x], [node], color=plot_utils.colors[0])

        if isinstance(layer, (MaxOut, ChannelOut)):
            group_nodes(x, nodes, layer.num_groups, ax=ax)

        x += 1
    return ax


def plot_active_nodes(layers: list, isactive: list, ax=None):
    """Plots all the nodes in every layer in layers, grouping LWTA
    and indicating whether the nodes are active or not.

    Args:
        layers (list): list of tf.keras.layers.Layer instances.
        isactive (list): list of boolean np.ndarrays indicating whether node
                         is active or not by layer.
        ax (ax, optional): plt ax on which to plot. Defaults to None.

    Returns:
        tuple: ax object on which the function plotted.
    """
    if ax is None:
        _, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])

    x = 0  # Position to plot current layer
    for layer, active in zip(layers, isactive):
        colors = np.where(active, "r", "b")
        try:
            nodes = layer.units
        except AttributeError:
            continue
        for node, color in zip(np.arange(-nodes/2, nodes/2), colors):
            ax.scatter([x], [node], c=color)

        if isinstance(layer, (MaxOut, ChannelOut)):
            group_nodes(x, nodes, layer.num_groups, ax=ax)

        x += 1
    return ax


def plot_pathways(layers: list, isactive: list, ax=None, **plot_kwargs):
    """Plots lines going through all active nodes of all layers.

    Args:
        layers (list): list of tf.keras.layers.Layer instances.
        isactive (list): list of boolean np.ndarrays indicating whether node
                         is active or not by layer.
        ax (ax, optional): plt ax on which to plot. Defaults to None.

    Returns:
        tuple: ax object on which the function plotted.
    """
    if ax is None:
        _, ax = plt.subplots()
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])

    line_kwargs = dict(
        color=plot_utils.colors[1],
        lw=1,
        alpha=0.05
    )
    line_kwargs.update(plot_kwargs)

    active_nodes_old = [np.nan]

    x = 0  # Position to plot current layer
    for layer, active in zip(layers, isactive):
        try:
            nodes = layer.units
        except AttributeError:
            continue
        active_nodes_new = np.where(active,
                                    np.arange(-nodes/2, nodes/2),
                                    np.nan)
        for old_node in active_nodes_old:
            for new_node in active_nodes_new:
                ax.plot([x-1, x], [old_node, new_node], **line_kwargs)
        active_nodes_old = active_nodes_new

        x += 1
    return ax
