"""
Tests with a single nodal point.
"""

# pylint: disable=redefined-outer-name,unused-argument

import tempfile

import pytest
import numpy as np
import scipy.linalg as la

from nodefinder import NodeFinder


@pytest.fixture
def node_position():
    return [0.5] * 3


@pytest.fixture
def gap_fct(node_position):
    def inner(x):
        return la.norm(np.array(x) - node_position)

    return inner


@pytest.fixture
def check_result(node_position):
    def inner(result):
        nodes = result.nodal_points
        assert len(nodes) == 1
        node = nodes[0]
        print(node)
        assert np.isclose(node.gap, 0)
        assert np.allclose(node.k, node_position)

    return inner


def test_single_node(gap_fct, node_position, check_result):
    """
    Test that a single nodal point is found.
    """
    node_finder = NodeFinder(gap_fct=gap_fct, fct_listable=False)
    check_result(node_finder.run())


def test_save(gap_fct, node_position, check_result):
    """
    Test saving to a file
    """
    with tempfile.NamedTemporaryFile() as named_file:
        node_finder = NodeFinder(
            gap_fct=gap_fct, fct_listable=False, save_file=named_file.name
        )
        check_result(node_finder.run())


def test_restart(gap_fct, node_position, check_result):
    """
    Test that the calculation is done when restarting from a finished result.
    """

    def invalid_gap_fct(x):
        raise ValueError

    with tempfile.NamedTemporaryFile() as named_file:
        node_finder = NodeFinder(
            gap_fct=gap_fct, fct_listable=False, save_file=named_file.name
        )
        result = node_finder.run()
        check_result(result)

        restart_node_finder = NodeFinder(
            gap_fct=invalid_gap_fct, save_file=named_file.name, load=True
        )
        restart_result = restart_node_finder.run()
        check_result(restart_result)
