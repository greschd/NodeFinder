# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines functions for plotting the results of the identify step.
"""

from functools import singledispatch

import numpy as np
import scipy.linalg as la
from fsc.export import export

from .result import NodalPoint, NodalLine

from .._common_plot import _setup_plot


@export
def result(res, *, axis=None):
    """Plot the result of the identify step.

    Arguments
    ---------
    res : IdentificationResultContainer
        Result of the identify step.
    axis : matplotlib.axes.Axes, optional
        Axes on which the result is plotted.
    """
    fig, axis, _ = _setup_plot(res.coordinate_system.limits, axis=axis)
    feature_size = res.feature_size
    for identification_result in res:
        shape = identification_result.shape
        color = axis._get_lines.get_next_color()  # pylint: disable=protected-access
        if shape is None:
            _plot_positions(identification_result.positions, axis=axis, color=color)
        else:
            _plot_result(shape, axis=axis, color=color, feature_size=feature_size)
    return fig, axis


def _plot_positions(positions, *, axis, color):
    coordinates = list(np.array(list(positions)).T)
    axis.scatter(*coordinates, color=color)


@singledispatch
def _plot_result(shape, axis, color, feature_size):
    raise NotImplementedError


@export
@_plot_result.register(NodalPoint)
def nodal_point(shape, *, axis, color, feature_size=None):
    """
    Plot a nodal point.

    Arguments
    ---------
    shape : NodalPoint
        Nodal point to be plotted.
    axis : matplotlib.axes.Axes
        Axes on which to plot.
    color : str
        Color of the point.
    feature_size : float
        Distance between two nodal points at which they are considered distinct.
        This argument is not used in this function.
    """
    coordinates = [[val] for val in shape.position]
    axis.scatter(*coordinates, color=color)


@export
@_plot_result.register(NodalLine)
def nodal_line(shape, *, axis, color, feature_size=None):
    """
    Plot a nodal line.

    Arguments
    ---------
    shape : NodalLine
        Nodal line to be plotted.
    axis : matplotlib.axes.Axes
        Axes on which to plot.
    color : str
        Color of the nodal line.
    feature_size : float
        Distance between two nodal points at which they are considered distinct.
        Used for cutting the line when it goes across periodic boundaries.
    """
    if feature_size is None:
        feature_size = np.inf

    graph = shape.graph

    paths = _get_graph_paths(graph, feature_size=feature_size)

    if paths:
        for path in paths:
            axis.plot(*np.array(path).T, color=color)  # pylint: disable=not-an-iterable
    else:
        axis.scatter(*np.array(list(graph.nodes)).T, color=color)  # pylint: disable=not-an-iterable


def _get_graph_paths(graph, feature_size):
    """
    Separate a graph into paths, breaking when there is no neighbor or when
    passing across the periodic boundary.
    """
    working_graph = graph.copy()

    paths = []
    while working_graph.edges:
        curr_node = _get_next_starting_point(working_graph)
        curr_path = [curr_node]
        while True:
            try:
                next_node = next(working_graph.neighbors(curr_node))
            except StopIteration:
                paths.append(curr_path)
                break
            if la.norm(np.array(next_node) - np.array(curr_node)) > 2 * feature_size:
                paths.append(curr_path)
                curr_path = [next_node]
            else:
                curr_path.append(next_node)

            working_graph.remove_edge(curr_node, next_node)
            curr_node = next_node
    return paths


def _get_next_starting_point(graph):
    nonzero_degree = [(node, degree) for node, degree in graph.degree if degree > 0]
    return min(nonzero_degree, key=lambda val: val[1] if val[1] != 2 else float("inf"))[
        0
    ]
