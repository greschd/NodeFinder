"""
Defines the container classes for object shapes.
"""

from types import SimpleNamespace

from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5


@export
@subscribe_hdf5('nodefinder.nodal_point')
class NodalPoint(SimpleNamespace, SimpleHDF5Mapping):
    """
    Shape class defining a nodal point.

    Attributes
    ----------
    position : tuple(float)
        The position of the point.
    """
    HDF5_ATTRIBUTES = ['position']

    def __init__(self, position):
        self.position = position


@export
@subscribe_hdf5('nodefinder.nodal_line')
class NodalLine(SimpleNamespace, SimpleHDF5Mapping):
    """
    Shape class defining a closed nodal line.

    Attributes
    ----------
    path : list(tuple(float))
        A list of positions describing the line.
    """
    HDF5_ATTRIBUTES = ['path']

    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return 'NodalLine(path=<{} values>)'.format(len(self.path))
