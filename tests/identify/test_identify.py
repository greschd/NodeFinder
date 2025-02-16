# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the identify step.
"""

import os

import pytest
import numpy as np

import nodefinder as nf


@pytest.fixture
def run_identify(sample):
    """
    Fixture to run the identification step for a given sample file.
    """

    def inner(sample_name):
        search_res = nf.io.load(sample(os.path.join("search", sample_name)))
        return nf.identify.run(search_res)

    return inner


@pytest.fixture
def run_single_identify(run_identify):  # pylint: disable=redefined-outer-name
    """
    Fixture to run the identification step for a given sample file which should
    contain only one cluster, and return the result for that cluster.
    """

    def inner(sample_name):
        res = run_identify(sample_name)
        assert len(res) == 1
        return res[0]

    return inner


def test_point(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that a point is correctly identified.
    """
    res = run_single_identify("point.hdf5")
    assert res.dimension == 0
    assert np.allclose(res.shape.position, [0.2, 0.4, 0.8])


def test_line(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that a line is correctly identified.
    """
    res = run_single_identify("line.hdf5")
    assert res.dimension == 1
    assert len(res.shape.graph.nodes) > 10
    assert res.shape.degree_count == dict()


def test_surface(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that a surface is correctly identified.
    """
    res = run_single_identify("surface.hdf5")
    assert res.dimension == 2


def test_two_lines(run_identify):  # pylint: disable=redefined-outer-name
    """
    Test that two 2D lines are correctly identified.
    """
    res = run_identify("two_lines.hdf5")
    assert len(res) == 2
    for identified_object in res:
        assert identified_object.shape.degree_count == dict()
        assert identified_object.dimension == 1
        assert len(identified_object.shape.graph.nodes) > 10


def test_cross(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that the cross is identified without holes.
    """
    res = run_single_identify("cross.hdf5")
    assert res.dimension == 1
    assert res.shape.degree_count == {4: 1}


def test_open_line(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that an open line is identified correctly.
    """
    res = run_single_identify("line_open.hdf5")
    assert res.dimension == 1
    assert res.shape.degree_count == {1: 2}
    positions = np.array(list(res.shape.graph.nodes))
    assert np.min(positions) < -0.49
    assert np.max(positions) > 0.49


def test_line_short_paths(run_single_identify):  # pylint: disable=redefined-outer-name
    """
    Test that a short line is identified correctly, without holes.
    """
    res = run_single_identify("line_1d.hdf5")
    assert res.dimension == 1
    assert res.shape.degree_count == dict()
