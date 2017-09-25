"""
Implements the node finding algorithm.
"""

import itertools
from types import SimpleNamespace

import numpy as np
from ._nelder_mead import root_nelder_mead


class NodalPoint(SimpleNamespace):
    """
    Result class for a nodal point.
    """

    def __init__(self, k, gap):
        super().__init__()
        self.k = tuple(np.array(k) % 1)
        self.gap = gap


class NodeFinder:
    """
    :param gap_fct: Function that returns the gap, given a k-point.

    :param gap_threshold: Threshold when the gap is considered to be closed.
    :type gap_threshold: float

    :param mesh_size: Initial mesh of starting points.
    :type mesh_size: tuple[int]

    :param feature_size: Minimum distance between nodal features for them to be considered distinct.
    :type feature_size: float
    """

    def __init__(
        self,
        gap_fct,
        *,
        gap_threshold=1e-6,
        mesh_size=(10, 10, 10),
        feature_size=1e-3
    ):
        self.gap_fct = gap_fct
        self._gap_threshold = gap_threshold
        self._mesh_size = tuple(mesh_size)
        self._feature_size = feature_size
        self._nodal_points = []
        self._initialize()

    def _initialize(self):
        self._calculate_box(
            box_position=((0, 1), ) * 3,
            mesh_size=self._mesh_size,
            periodic=True
        )

    def _calculate_box(self, *, box_position, mesh_size, periodic=False):
        """
        Search for minima, with starting positions in a mesh of a given box.

        :param box_position: Boundaries of the box, given as list of tuples, e.g. [(min_x, max_x), (min_y, max_y), (min_z, max_z)].
        :type box_position: list[tuple[float]]
        """
        mesh = itertools.product(
            *[
                np.linspace(min_val, max_val, N, endpoint=not periodic)
                for (min_val, max_val), N in zip(box_position, mesh_size)
            ]
        )
        for starting_point in mesh:
            trial_point = self._minimize(starting_point=starting_point)
            if trial_point.fun < self._gap_threshold:
                self._nodal_points.append(
                    NodalPoint(k=trial_point.x, gap=trial_point.fun)
                )

    def _minimize(self, starting_point):
        """
        Minimize the gap function from the given starting point.
        """
        # TODO:
        # * Change the minimization to contain the dynamic cutoff criterion
        # * Make cutoff values configurable
        # * Allow setting the other starting vertices of the Nelder-Mead algorithm
        # res = so.minimize(self.gap_fct, x0=starting_point, method='Nelder-Mead', tol=1e-8, options=dict(maxfev=20))
        # if res.fun < 0.1:
        #     res = so.minimize(self.gap_fct, x0=res.x, method='Nelder-Mead', tol=1e-8, options=dict(maxfev=100))
        #     if res.fun < 1e-2:
        res = root_nelder_mead(self.gap_fct, x0=starting_point)
        # print(res)
        return res
