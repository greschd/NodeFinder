import itertools
from types import MappingProxyType

import numpy as np

from .._common_plot import _setup_plot


def points(result, *, axis=None):
    fig, axis, is_3d = _setup_plot(result.coordinate_system.limits, axis=axis)
    x_coords = []
    y_coords = []
    if is_3d:
        z_coords = []
    vals = []
    for node in result.nodes:
        pos = node.pos
        x, y, z = pos
        x_coords.append(x)
        y_coords.append(y)
        if is_3d:
            z_coords.append(z)
        vals.append(node.value)

    if is_3d:
        coords = [x_coords, y_coords, z_coords]
    else:
        coords = [x_coords, y_coords]

    axis.scatter(*coords, c=vals)
    return fig, axis


def simplices(
    result,
    *,
    nodes=(),
    axis=None,
    line_settings=MappingProxyType(dict(color='C0'))
):
    fig, axis, _ = _setup_plot(result.coordinate_system.limits, axis=axis)
    for node in nodes:
        for simplex in node.simplex_history:
            _plot_simplex(axis=axis, simplex=simplex, **line_settings)
    return fig, axis


def _plot_simplex(axis, simplex, **kwargs):
    for start, end in itertools.combinations(simplex, 2):
        values = list(np.array([start, end]).T)
        axis.plot(*values, **kwargs)
