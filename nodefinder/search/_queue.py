"""
Defines the SimplexQueue, which tracks the state of simplices to be minimized.
"""

from queue import Queue

import numpy as np
from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, from_hdf5


@export
@subscribe_hdf5('nodefinder.simplex_queue')
class SimplexQueue(HDF5Enabled):
    """
    Contains the running and queued initial simplices during a search calculation.
    """

    def __init__(self, simplices=frozenset()):
        all_simplices = self.convert_to_sorted_tuples(simplices)
        self._queued_simplices = Queue()
        self._extend_queue(all_simplices)
        self._running_simplices = set()
        self._all_simplices = set(all_simplices)
        self.needs_saving = True

    @staticmethod
    def convert_to_sorted_tuples(simplices):
        return [
            tuple(sorted(tuple(coord) for coord in simplex))
            for simplex in simplices
        ]

    @property
    def simplices(self):
        # Give running simplices first, so that they will be re-queued first when
        # restarting a calculation.
        return list(self._running_simplices
                    ) + list(self._queued_simplices.queue)

    def pop_queued(self):
        """
        Get a queued simplex, and add it to the running simplices.
        """
        starting_point = self._queued_simplices.get_nowait()
        self._running_simplices.add(starting_point)
        return starting_point

    def add_simplices(self, simplices):
        """
        Add new simplices to the queue.
        """
        new_simplices = self.convert_to_sorted_tuples(simplices)
        new_simplices_filtered = [
            simplex for simplex in new_simplices
            if simplex not in self._all_simplices
        ]
        self._extend_queue(new_simplices_filtered)
        self._all_simplices.update(new_simplices_filtered)
        self.needs_saving = True

    def _extend_queue(self, simplices):
        """
        Add given simplices to '_queued_simplices'. Note that this does _not_
        handle the other attributes, use 'add_simplices' for this purpose.
        """
        for simplex in simplices:
            self._queued_simplices.put_nowait(simplex)

    def set_finished(self, simplex):
        """
        Mark a given simplex as finished.
        """
        self._running_simplices.remove(simplex)
        self.needs_saving = True

    @property
    def has_queued_points(self):
        """
        Shows if there are currently queued simplices.
        """
        return not self._queued_simplices.empty()

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
        simplices_array = np.array(self.simplices)
        hdf5_handle['simplices_array'] = simplices_array

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        try:
            simplices_array = np.array(hdf5_handle['simplices_array'])
            simplices = [
                tuple(tuple(pos) for pos in simp) for simp in simplices_array
            ]
        # handle old data version
        except KeyError:
            simplices = from_hdf5(hdf5_handle['simplices'])
        return cls(simplices=simplices)
