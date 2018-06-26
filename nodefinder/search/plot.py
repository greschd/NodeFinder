import itertools
from types import MappingProxyType

import decorator
import numpy as np


def _plot(proj_3d=False):
    """Decorator that sets up the figure axes and handles options common to all plots."""

    @decorator.decorator
    def inner(func, data, *, axis=None, **kwargs):  # pylint: disable=missing-docstring
        # import is here s.t. the import of the package does not fail
        # if matplotlib is not present
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

        # create axis if it does not exist
        if axis is None:
            return_fig_ax = True
            fig = plt.figure()
            axis = fig.add_subplot(111, projection='3d' if proj_3d else None)
        else:
            return_fig_ax = False

        limits = data.coordinate_system.limits

        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        if proj_3d:
            axis.set_zlim(*limits[2])

        func(data, axis=axis, **kwargs)

        if return_fig_ax:
            return fig, axis

    return inner


@_plot(proj_3d=True)
def plot_3d(result, *, axis=None):  # pylint: disable=inconsistent-return-statements
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


@_plot(proj_3d=True)
def plot_simplices_3d(
    result,  # pylint: disable=unused-argument
    *,
    nodes=(),
    axis=None,
    line_settings=MappingProxyType(dict(color='C0'))
):
    for node in nodes:
        for simplex in node.simplex_history:
            _plot_simplex(axis=axis, simplex=simplex, **line_settings)


def _plot_simplex(axis, simplex, **kwargs):
    for start, end in itertools.combinations(simplex, 2):
        x_values, y_values, z_values = np.array([start, end]).T
        axis.plot(x_values, y_values, z_values, **kwargs)
