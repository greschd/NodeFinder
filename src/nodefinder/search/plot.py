# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines functions for plotting the results of the search step.
"""

import itertools
from types import MappingProxyType

import numpy as np
from fsc.export import export

from .._common_plot import _setup_plot


@export
def points(result, *, axis=None):
    """
    Plot the nodal points of a given result.

    Arguments
    ---------
    result : SearchResultContainer
        Result whose nodal points should be plotted.
    axis : matplotlib.axes.Axes, optional
        Axes on which the points should be plotted.
    """
    fig, axis, is_3d = _setup_plot(result.coordinate_system.limits, axis=axis)
    x_coords = []
    y_coords = []
    if is_3d:
        z_coords = []
    vals = []
    for node in result.nodes:
        pos = node.pos
        x, y, *z = pos
        x_coords.append(x)
        y_coords.append(y)
        if is_3d:
            z_coords.append(z[0])
        vals.append(node.value)

    if is_3d:
        coords = [x_coords, y_coords, z_coords]
    else:
        coords = [x_coords, y_coords]

    axis.scatter(*coords, c=vals)
    return fig, axis


@export
def simplices(
    result, *, nodes=(), axis=None, line_settings=MappingProxyType(dict(color="C0"))
):
    """
    Plot the simplices used in the minimization for a given node.

    Arguments
    ---------
    result : SearchResultContainer
        Result of the search step.
    nodes : list(MinimizationResult)
        Nodes for which the simplex history should be plotted.
    axis : matplotlib.axes.Axes, optional
        Axes on which the points should be plotted.
    line_settings : dict
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    fig, axis, _ = _setup_plot(result.coordinate_system.limits, axis=axis)
    for node in nodes:
        for simplex in node.simplex_history:
            _plot_simplex(axis=axis, simplex=simplex, **line_settings)
    return fig, axis


@export
def stencil(
    stcl,
    *,
    axis=None,
    line_settings=MappingProxyType(dict(color="C0")),
    origin_settings=MappingProxyType(dict(color="C1")),
    plot_sphere=True,
    sphere_settings=MappingProxyType(dict(color="C2", alpha=0.2)),
):
    """
    Plot a refinement stencil.
    """
    dim = stcl.shape[-1]
    vertices = stcl.reshape(-1, dim)
    limits = list(zip(np.min(vertices, axis=0), np.max(vertices, axis=0)))
    fig, axis, _ = _setup_plot(limits, axis=axis)
    for simplex in stcl:
        _plot_simplex(axis=axis, simplex=simplex, **line_settings)
    if origin_settings is not None:
        axis.scatter(*np.zeros(dim), **origin_settings)
    if plot_sphere:
        if dim == 2:
            phi = np.linspace(0, 2 * np.pi, 200)
            axis.plot(np.cos(phi), np.sin(phi), **sphere_settings)
        elif dim == 3:
            phi = np.linspace(0, 2 * np.pi, 200)
            theta = np.linspace(0, np.pi, 200)
            x = np.outer(np.cos(phi), np.sin(theta))
            y = np.outer(np.sin(phi), np.sin(theta))
            z = np.outer(np.ones_like(phi), np.cos(theta))
            axis.plot_surface(x, y, z, **sphere_settings)
    return fig, axis


def _plot_simplex(axis, simplex, **kwargs):
    for start, end in itertools.combinations(simplex, 2):
        values = list(np.array([start, end]).T)
        axis.plot(*values, **kwargs)
