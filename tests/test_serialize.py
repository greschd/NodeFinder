"""
Test serialization to HDF5.
"""

# pylint: disable=redefined-outer-name

import tempfile

import pytest
from fsc.hdf5_io import save, load

from nodefinder._result import NodalPoint, StartingPoint, NodeFinderResult


@pytest.fixture
def save_load():
    def inner(x):
        with tempfile.NamedTemporaryFile() as named_file:
            save(x, named_file.name)
            return load(named_file.name)

    return inner


def test_nodal_point(save_load):
    """
    Test saving a single NodalPoint.
    """
    point = NodalPoint(k=(1.5, 0.9, 0.2), gap=1e-3)
    pt_copy = save_load(point)
    assert point == pt_copy


def test_starting_point(save_load):
    """
    Test saving a StartingPoint.
    """
    point = StartingPoint(k=(0.2, 0.9, 1.1))
    pt_copy = save_load(point)
    assert point == pt_copy


def test_result(save_load):
    """
    Test saving a NodeFinderResult.
    """
    starting_points = [
        StartingPoint(k=(0.1, 0.5, 0.2)),
        StartingPoint(k=(0.1, 0.3, 0.2)),
    ]
    result = NodeFinderResult(
        gap_threshold=1e-4, feature_size=1e-2, starting_points=starting_points
    )
    running_pt = result.pop_queued_starting_point()
    result.add_result(
        starting_point=running_pt,
        nodal_point=NodalPoint(k=(1.5, 0.9, 0.2), gap=1e-5)
    )
    result_copy = save_load(result)
    # pylint: disable=protected-access
    assert result_copy.nodal_points == result.nodal_points
    assert result_copy.starting_points == result.starting_points
    assert result_copy._gap_threshold == result._gap_threshold
    assert result_copy._feature_size == result._feature_size
