"""
Implements the node finding algorithm.
"""

import copy
import asyncio
import itertools
from types import SimpleNamespace

import numpy as np
import scipy.linalg as la

from ._logging import _LOGGER
from ._nelder_mead import root_nelder_mead
from ._batch_submit import BatchSubmitter


class NodalPoint(SimpleNamespace):
    """
    Result class for a nodal point.
    """

    def __init__(self, k, gap):
        super().__init__()
        self.k = tuple(np.array(k) % 1)
        self.gap = gap


class NodalPointContainer:
    def __init__(self, *, feature_size, gap_threshold):
        self._feature_size = feature_size
        self._gap_threshold = gap_threshold
        self._nodal_points = []
        self._new_points = []

    def add(self, nodal_point):
        if nodal_point.gap < self._gap_threshold:
            k = np.array(nodal_point.k)
            if all(
                periodic_distance(k, n.k) > self._feature_size
                for n in self._nodal_points
            ):
                self._nodal_points.append(nodal_point)
                self._new_points.append(nodal_point)
                return True
        return False

    def get_new_points(self):
        return copy.copy(self._new_points)

    def clear_new_points(self):
        self._new_points = []

    def get_nodes(self):
        return copy.copy(self._nodal_points)


class NodeFinder:
    """
    :param gap_fct: Function that returns the gap, given a k-point.

    :param gap_threshold: Threshold when the gap is considered to be closed.
    :type gap_threshold: float

    :param feature_size: Minimum distance between nodal features for them to be considered distinct.
    :type feature_size: float

    :param initial_box_position: Initial box within which the minimization starting points are selected.
    :type initial_box_position: tuple(tuple(float))

    :param mesh_size: Initial mesh of starting points.
    :type mesh_size: tuple[int]
    """

    def __init__(
        self,
        gap_fct,
        *,
        fct_listable=True,
        gap_threshold=1e-6,
        feature_size=1e-3,
        initial_box_position=((0, 1), ) * 3,
        mesh_size=(10, 10, 10),
        refinement_box_size=5e-3,
        refinement_mesh_size=(2, 2, 2),
        **nelder_mead_kwargs
    ):
        if fct_listable:
            listable_gap_fct = gap_fct
        else:
            listable_gap_fct = lambda input_list: [gap_fct(x) for x in input_list]

        self._batch_submitter = BatchSubmitter(listable_gap_fct)
        self._func = self._batch_submitter.submit
        self._mesh_size = tuple(mesh_size)
        self._refinement_dist = refinement_box_size / 2
        self._refinement_mesh_size = refinement_mesh_size
        self._nodal_point_container = NodalPointContainer(
            feature_size=feature_size, gap_threshold=gap_threshold
        )
        self._nelder_mead_kwargs = nelder_mead_kwargs
        self._initial_box_position = initial_box_position

    def run(self):
        loop = asyncio.get_event_loop()
        with self._batch_submitter:
            loop.run_until_complete(self._run())

    async def _run(self):
        await self._calculate_box(
            box_position=self._initial_box_position,
            mesh_size=self._mesh_size,
            periodic=
            True  # TODO: Change this depending on the value of initial_box_position.
        )
        while True:
            new_points = self._nodal_point_container.get_new_points()
            self._nodal_point_container.clear_new_points()
            if not new_points:
                break
            _LOGGER.info('{num_pts} new points found', num_pts=len(new_points))
            await asyncio.gather(
                *[
                    self._calculate_box(
                        box_position=tuple((
                            ki - self._refinement_dist,
                            ki + self._refinement_dist
                        ) for ki in new_node.k),
                        mesh_size=self._refinement_mesh_size,
                        periodic=False
                    ) for new_node in new_points
                ]
            )

    async def _calculate_box(self, *, box_position, mesh_size, periodic=False):
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
        trial_points = await asyncio.gather(
            *[
                self._minimize(starting_point=starting_point)
                for starting_point in mesh
            ]
        )
        for point in trial_points:
            self._nodal_point_container.add(
                NodalPoint(k=point.x, gap=point.fun)
            )

    async def _minimize(self, starting_point):
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
        res = await root_nelder_mead(
            self._func, x0=starting_point, **self._nelder_mead_kwargs
        )
        return res

    @property
    def nodal_points(self):
        return self._nodal_point_container.get_nodes()


def periodic_distance(k1, k2):
    return la.norm([_periodic_distance_1d(a, b) for a, b in zip(k1, k2)])


def _periodic_distance_1d(k1, k2):
    k1 %= 1
    k2 %= 1
    return min((k1 - k2) % 1, (k2 - k1) % 1)
