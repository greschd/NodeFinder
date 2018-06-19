"""
Implements the node finding algorithm.
"""

import uuid
from collections import deque
from types import SimpleNamespace

import numpy as np
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5



@subscribe_hdf5('nodefinder.starting_point')
class StartingPoint(SimpleNamespace, HDF5Enabled):
    """
    Result class for a minimization starting point.
    """

    def __init__(self, k, id_=None):
        super().__init__()
        self.k = tuple(np.array(k) % 1)
        self.uuid = id_ or uuid.uuid4()

    def to_hdf5(self, hdf5_handle):
        hdf5_handle['k'] = self.k
        hdf5_handle['uuid'] = self.uuid.hex

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            k=hdf5_handle['k'].value,
            id_=uuid.UUID(hex=hdf5_handle['uuid'].value)
        )

    def __hash__(self):
        return self.uuid.int


@subscribe_hdf5('nodefinder.node_finder_result')
class NodeFinderResult(HDF5Enabled):
    def __init__(
        self,
        *,
        feature_size,
        gap_threshold,
        nodal_points=(),
        starting_points=frozenset()
    ):
        self._feature_size = feature_size
        self._gap_threshold = gap_threshold
        self.nodal_points = list(nodal_points)

    def add_result(self, starting_point, nodal_point):
        self._running_starting_points.remove(starting_point)

        if nodal_point.gap < self._gap_threshold:
            k = np.array(nodal_point.k)
            if all(
                periodic_distance(k, n.k) > self._feature_size
                for n in self.nodal_points
            ):
                self.nodal_points.append(nodal_point)
                return True
        return False

    def pop_queued_starting_point(self):
        starting_point = self._queued_starting_points.popleft()
        self._running_starting_points.add(starting_point)
        return starting_point

    def add_starting_points(self, starting_points):
        self._queued_starting_points.extend(starting_points)

    @property
    def starting_points(self):
        # Give running points first, so that they will be re-queued first when
        # restarting a calculation.
        return list(self._running_starting_points
                    ) + list(self._queued_starting_points)

    @property
    def num_running(self):
        return len(self._running_starting_points)


    def to_hdf5(self, hdf5_handle):
        nodal_points = hdf5_handle.create_group('nodal_points')
        to_hdf5(self.nodal_points, nodal_points)
        starting_points = hdf5_handle.create_group('starting_points')
        to_hdf5(self.starting_points, starting_points)
        hdf5_handle['feature_size'] = self._feature_size
        hdf5_handle['gap_threshold'] = self._gap_threshold

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            feature_size=hdf5_handle['feature_size'].value,
            gap_threshold=hdf5_handle['gap_threshold'].value,
            nodal_points=from_hdf5(hdf5_handle['nodal_points']),
            starting_points=from_hdf5(hdf5_handle['starting_points']),
        )
