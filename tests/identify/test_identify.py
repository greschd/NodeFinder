import os

import pytest
import numpy as np

import nodefinder as nf


@pytest.fixture
def run_identify(sample):
    def inner(sample_name):
        search_res = nf.io.load(sample(os.path.join('search', sample_name)))
        return nf.identify.run(search_res)

    return inner


@pytest.fixture
def run_single_identify(run_identify):  # pylint: disable=redefined-outer-name
    def inner(sample_name):
        res = run_identify(sample_name)
        assert len(res) == 1
        return res[0]

    return inner


def test_point(run_single_identify):  # pylint: disable=redefined-outer-name
    res = run_single_identify('point.hdf5')
    assert res.dimension == 0
    assert np.allclose(res.result.position, [0.2, 0.4, 0.8])


def test_line(run_single_identify):  # pylint: disable=redefined-outer-name
    res = run_single_identify('line.hdf5')
    assert res.dimension == 1
    assert len(res.result.path) > 10


def test_surface(run_single_identify):  # pylint: disable=redefined-outer-name
    res = run_single_identify('surface.hdf5')
    assert res.dimension == 2


def test_two_lines(run_identify):  # pylint: disable=redefined-outer-name
    res = run_identify('two_lines.hdf5')
    assert len(res) == 2
    for identified_object in res:
        assert identified_object.dimension == 1
        assert len(identified_object.result.path) > 10
