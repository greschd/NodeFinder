"""
Test serialization to HDF5.
"""

import tempfile

import pytest
from fsc.hdf5_io import save, load

from nodefinder._nodefinder import NodalPoint, NodalPointContainer


@pytest.fixture
def save_load():
    def inner(x):
        with tempfile.NamedTemporaryFile() as named_file:
            save(x, named_file.name)
            return load(named_file.name)

    return inner


def test_nodal_point(save_load):  # pylint: disable=redefined-outer-name
    """
    Test saving a single NodalPoint.
    """
    point = NodalPoint(k=(1.5, 0.9, 0.2), gap=1e-3)
    pt_copy = save_load(point)
    assert point == pt_copy


def test_nodal_point_container(save_load):  # pylint: disable=redefined-outer-name
    """
    Test saving a NodalPointContainer.
    """
    pt_container = NodalPointContainer(gap_threshold=1e-4, feature_size=1e-2)
    pt_container.add(NodalPoint(k=(1.5, 0.9, 0.2), gap=1e-5))
    pt_container.clear_new_points()
    pt_container.add(NodalPoint(k=(1.1, 0.8, 0.1), gap=1e-6))
    pt_container.add(NodalPoint(k=(1.1, 0.1, 0.1), gap=1e-6))
    container_copy = save_load(pt_container)
    # pylint: disable=protected-access
    assert container_copy._nodal_points == pt_container._nodal_points
    assert container_copy._new_points == pt_container._new_points
    assert container_copy._gap_threshold == pt_container._gap_threshold
    assert container_copy._feature_size == pt_container._feature_size
