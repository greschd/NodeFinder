from collections import deque

from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5


@export
@subscribe_hdf5('nodefinder.starting_point_queue')
class StartingPointQueue(HDF5Enabled):
    def __init__(self, starting_points=frozenset()):
        self._queued_starting_points = deque(starting_points)
        self._running_starting_points = set()

    @property
    def starting_points(self):
        # Give running points first, so that they will be re-queued first when
        # restarting a calculation.
        return list(self._running_starting_points
                    ) + list(self._queued_starting_points)

    def pop_queued(self):
        starting_point = self._queued_starting_points.popleft()
        self._running_starting_points.add(starting_point)
        return starting_point

    def add_starting_points(self, starting_points):
        self._queued_starting_points.extend(starting_points)

    def set_finished(self, starting_point):
        self._running_starting_points.remove(starting_point)

    @property
    def has_queued_points(self):
        return bool(self._queued_starting_points)

    @property
    def finished(self):
        return not self.starting_points

    @property
    def num_running(self):
        return len(self._running_starting_points)

    def to_hdf5(self, hdf5_handle):
        starting_points = hdf5_handle.create_group('starting_points')
        to_hdf5(self.starting_points, starting_points)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(starting_points=from_hdf5(hdf5_handle['starting_points']), )
