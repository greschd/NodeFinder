"""
Defines the container classes for the identification results.
"""

from types import SimpleNamespace

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping


@export
@subscribe_hdf5('nodefinder.identification_result_container')
class IdentificationResultContainer(SimpleNamespace, SimpleHDF5Mapping):
    """Container class for the result of the identification step.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system of the problem.
    results : list(IdentificationResult)
        List of identified objects.
    feature_size : float
        The ``feature_size`` used when identifying the objects.
    """
    HDF5_ATTRIBUTES = ['coordinate_system', 'results', 'feature_size']

    def __init__(self, *, coordinate_system, feature_size, results=()):
        self.coordinate_system = coordinate_system
        self.results = results
        self.feature_size = feature_size

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    def __len__(self):
        return len(self.results)


@export
@subscribe_hdf5('nodefinder.identification_result')
class IdentificationResult(SimpleNamespace, SimpleHDF5Mapping):
    """Contains the attributes of an identified object.

    Attributes
    ----------
    positions : list(tuple(float))
        Positions of the nodal points making up the object.
    shape : :obj:`None` or NodalPoint or NodalLine
        Shape of the identified object. If the shape could not be identified, it
        is set to ``None``.
    dimension : int
        Dimension of the identified object. Is set to ``None`` if the dimension
        is ambiguous.
    """
    HDF5_ATTRIBUTES = ['positions', 'shape', 'dimension']

    def __init__(self, positions, dimension, shape=None):
        self.positions = positions
        self.dimension = dimension
        self.shape = shape

    def __repr__(self):
        return 'IdentificationResult(dimension={}, shape={}, positions=<{} values>)'.format(
            self.dimension, self.shape, len(self.positions)
        )
