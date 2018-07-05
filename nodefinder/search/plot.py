import itertools
from types import MappingProxyType

import numpy as np

from .._common_plot import _setup_plot


def points_3d(result, *, axis=None):
    fig, axis = _setup_plot(result.coordinate_system.limits, axis=axis)
    x_coords = []
    y_coords = []
    z_coords = []
    vals = []
    for node in result.nodes:
        pos = node.pos
        x, y, z = pos
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        vals.append(node.value)

    axis.scatter(x_coords, y_coords, z_coords, c=vals)
    return fig, axis


def simplices_3d(
    result,  # pylint: disable=unused-argument
    *,
    nodes=(),
    axis=None,
    line_settings=MappingProxyType(dict(color='C0'))
):
    fig, axis = _setup_plot(result.coordinate_system.limits, axis=axis)
    for node in nodes:
        for simplex in node.simplex_history:
            _plot_simplex(axis=axis, simplex=simplex, **line_settings)
    return fig, axis


def _plot_simplex(axis, simplex, **kwargs):
    for start, end in itertools.combinations(simplex, 2):
        x_values, y_values, z_values = np.array([start, end]).T
        axis.plot(x_values, y_values, z_values, **kwargs)
