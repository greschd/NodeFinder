# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the SimplexQueue, which tracks the state of simplices to be minimized.
"""

from abc import ABC, abstractmethod
from queue import Queue

import numpy as np
from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5


class ObjectQueue(HDF5Enabled, ABC):
    """
    General queue class. Implements caching (queueing objects only once) and
    HDF5 serialization on top of the built-in Queue.
    """

    HDF5_ATTRIBUTES = ["objects"]

    def __init__(self, objects=frozenset()):
        all_objects = self.normalize(objects)
        self._queued_objects = Queue()
        self._extend_queue(all_objects)

        self._all_objects = set(all_objects)
        self.needs_saving = True

    @abstractmethod
    def normalize(self, objects):
        raise NotImplementedError

    @property
    def objects(self):
        return list(self._queued_objects.queue)

    def pop_queued(self):
        return self._queued_objects.get_nowait()

    def add_objects(self, objects):
        """
        Add new objects to the queue.
        """
        new_objects = self.normalize(objects)
        new_objects_filtered = [
            obj for obj in new_objects if obj not in self._all_objects
        ]
        if new_objects_filtered:
            self._extend_queue(new_objects_filtered)
            self._all_objects.update(new_objects_filtered)
            self.needs_saving = True

    def _extend_queue(self, objects):
        """
        Add given objects to '_queued_objects'. Note that this does _not_
        handle the other attributes, use 'add_objects' for this purpose.
        """
        for obj in objects:
            self._queued_objects.put_nowait(obj)

    @property
    def has_queued(self):
        """
        Shows if there are currently queued objects.
        """
        return not self._queued_objects.empty()

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        # try:
        objects = np.array(hdf5_handle["objects"])
        # except
        return cls(objects=objects)

    def to_hdf5(self, hdf5_handle):
        objects = np.array(self.objects)
        hdf5_handle["objects"] = objects


class RunningQueue(ObjectQueue, HDF5Enabled):  # pylint: disable=abstract-method
    """
    Queue class for objects which can have a 'running' state. Objects are
    automatically put in the 'running' state when pop-ed from the queue, and
    need to be set to 'finished' to be removed from the queue. When reloading
    the queue, all running objects are put back into the queue.
    """

    def __init__(self, objects=frozenset()):
        super().__init__(objects=objects)
        self._running_objects = set()

    @property
    def objects(self):
        # Give running objects first, so that they will be re-queued first when
        # restarting a calculation.
        return list(self._running_objects) + super().objects

    def pop_queued(self):
        """
        Get a queued object, and add it to the running objects.
        """
        obj = super().pop_queued()
        self._running_objects.add(obj)
        return obj

    def set_finished(self, obj):
        """
        Mark a given object as finished.
        """
        self._running_objects.remove(obj)
        self.needs_saving = True

    @property
    def finished(self):
        """
        Indicates whether the queue is finished.
        """
        return not self.objects

    @property
    def num_running(self):
        """
        Gives the number of currently running objects.
        """
        return len(self._running_objects)


@export
@subscribe_hdf5("nodefinder.simplex_queue")
class SimplexQueue(RunningQueue):
    """
    Queue class for the simplices which should be minimized.
    """

    @staticmethod
    def normalize(objects):
        return [tuple(sorted(tuple(coord) for coord in simplex)) for simplex in objects]


@export
@subscribe_hdf5("nodefinder.position_queue")
class PositionQueue(ObjectQueue):
    """
    Queue class for the positions which should be refined.
    """

    @staticmethod
    def normalize(objects):
        return [tuple(pos) for pos in objects]
