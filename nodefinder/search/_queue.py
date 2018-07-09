"""
Defines the SimplexQueue, which tracks the state of simplices to be minimized.
"""

from collections import deque

from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5


@export
@subscribe_hdf5('nodefinder.simplex_queue')
class SimplexQueue(HDF5Enabled):
    """
    Contains the running and queued initial simplices during a search calculation.
    """

    def __init__(self, simplices=frozenset()):
        self._queued_simplices = deque(self.convert_to_tuples(simplices))
        self._running_simplices = set()
        self._all_simplices = set(self._queued_simplices)

    @staticmethod
    def convert_to_tuples(simplices):
        return [
            tuple(tuple(coord) for coord in simplex) for simplex in simplices
        ]

    @property
    def simplices(self):
        # Give running simplices first, so that they will be re-queued first when
        # restarting a calculation.
        return list(self._running_simplices) + list(self._queued_simplices)

    def pop_queued(self):
        """
        Get a queued simplex, and add it to the running simplices.
        """
        starting_point = self._queued_simplices.popleft()
        self._running_simplices.add(starting_point)
        return starting_point

    def add_simplices(self, simplices):
        """
        Add new simplices to the queue.
        """
        new_simplices = self.convert_to_tuples(simplices)
        new_simplices_filtered = [
            simplex for simplex in new_simplices
            if simplex not in self._all_simplices
        ]
        self._queued_simplices.extend(new_simplices_filtered)
        self._all_simplices.update(new_simplices_filtered)

    def set_finished(self, simplex):
        """
        Mark a given simplex as finished.
        """
        self._running_simplices.remove(simplex)

    @property
    def has_queued_points(self):
        """
        Shows if there are currently queued simplices.
        """
        return bool(self._queued_simplices)

    @property
    def finished(self):
        """
        Indicates whether the search calculation is done.
        """
        return not self.simplices

    @property
    def num_running(self):
        """
        Gives the number of currently running simplex minimizations.
        """
        return len(self._running_simplices)

    def to_hdf5(self, hdf5_handle):
        simplices = hdf5_handle.create_group('simplices')
        to_hdf5(self.simplices, simplices)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(simplices=from_hdf5(hdf5_handle['simplices']))
