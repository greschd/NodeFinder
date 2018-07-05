from functools import singledispatch

import numpy as np

from ._evaluate import NodalPoint, NodalLine

from .._common_plot import _setup_plot


def result(res, *, axis=None):
    fig, axis, _ = _setup_plot(res.coordinate_system.limits, axis=axis)
    for identification_result in res:
        res_identified = identification_result.result
        color = axis._get_lines.get_next_color()  # pylint: disable=protected-access
        if res_identified is None:
            _plot_positions(
                identification_result.positions, axis=axis, color=color
            )
        else:
            _plot_result(res_identified, axis=axis, color=color)
    return fig, axis


def _plot_positions(positions, *, axis, color):
    coordinates = list(np.array(list(positions)).T)
    axis.scatter(*coordinates, color=color)


@singledispatch
def _plot_result(res, axis, color):  # pylint: disable=unused-argument
    raise NotImplementedError


@_plot_result.register(NodalPoint)
def nodal_point(res, *, axis, color):
    coordinates = [[val] for val in res.position]
    axis.scatter(*coordinates, color=color)


@_plot_result.register(NodalLine)
def nodal_line(res, *, axis, color):
    axis.plot(*np.array(res.path).T, color=color)
