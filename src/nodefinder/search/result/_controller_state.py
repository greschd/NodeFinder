# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the data class for the controller state.
"""

from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5


@export
@subscribe_hdf5("nodefinder.controller_state")
class ControllerState(SimpleHDF5Mapping):
    """
    Container class for the current result and queue of the :func:`.search.run`
    function.
    """

    HDF5_ATTRIBUTES = ["result", "simplex_queue", "position_queue"]

    def __init__(self, *, result, simplex_queue, position_queue):
        self.result = result
        self.simplex_queue = simplex_queue
        self.position_queue = position_queue

    @property
    def needs_saving(self):
        return (
            self.result.needs_saving
            or self.simplex_queue.needs_saving
            or self.position_queue.needs_saving
        )

    @needs_saving.setter
    def needs_saving(self, value):
        assert not value
        self.result.needs_saving = value
        self.simplex_queue.needs_saving = value
        self.position_queue.needs_saving = value
