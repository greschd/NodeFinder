# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the container classes for the identification results.
"""

from types import SimpleNamespace

import numpy as np
from fsc.export import export
from fsc.hdf5_io import (
    subscribe_hdf5,
    SimpleHDF5Mapping,
    HDF5Enabled,
    to_hdf5,
    from_hdf5,
)


@export
@subscribe_hdf5("nodefinder.identification_result_container")
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

    HDF5_ATTRIBUTES = ["coordinate_system", "results", "feature_size"]

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
@subscribe_hdf5("nodefinder.identification_result")
class IdentificationResult(SimpleNamespace, HDF5Enabled):
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

    # HDF5_ATTRIBUTES = ['positions', 'shape', 'dimension']

    def __init__(self, positions, dimension, shape=None):
        self.positions = [tuple(pos) for pos in positions]
        self.dimension = dimension
        self.shape = shape

    def __repr__(self):
        return "IdentificationResult(dimension={}, shape={}, positions=<{} values>)".format(
            self.dimension, self.shape, len(self.positions)
        )

    def to_hdf5(self, hdf5_handle):
        to_hdf5(self.dimension, hdf5_handle.create_group("dimension"))
        hdf5_handle["positions"] = np.array(self.positions)
        to_hdf5(self.shape, hdf5_handle.create_group("shape"))

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        shape = from_hdf5(hdf5_handle["shape"])
        try:
            dimension = hdf5_handle["dimension"][()]
        except (AttributeError, TypeError):
            dimension = from_hdf5(hdf5_handle["dimension"])
        try:
            positions = [tuple(x) for x in hdf5_handle["positions"][()]]
        except (AttributeError, TypeError):
            positions = from_hdf5(hdf5_handle["positions"])
        return cls(positions=positions, dimension=dimension, shape=shape)
