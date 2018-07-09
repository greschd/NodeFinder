"""
Defines the data class for the controller state.
"""

from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5


@export
@subscribe_hdf5('nodefinder.controller_state')
class ControllerState(SimpleHDF5Mapping):
    """
    Container class for the current result and queue of the :func:`.search.run`
    function.
    """
    HDF5_ATTRIBUTES = ['result', 'queue']

    def __init__(self, *, result, queue):
        self.result = result
        self.queue = queue
