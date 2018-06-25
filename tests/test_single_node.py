"""
Tests with a single nodal point.
"""

# pylint: disable=redefined-outer-name,unused-argument

import tempfile

import pytest
import numpy as np
import scipy.linalg as la

from nodefinder import run_node_finder
from nodefinder._minimization._fake_potential import FakePotential

INITIAL_MESH_SIZE = (2, 2, 2)


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
    """
    Checks that the result matches a given node position.
    """

    def inner(result):
        for node in result.nodes:
            assert np.isclose(node.value, 0, atol=1e-6)
            assert np.allclose(node.pos, node_position)

    return inner


@pytest.fixture(params=[None, FakePotential])
def fake_potential_class(request):
    return request.param


def test_single_node(
    gap_fct,
    node_position,
    check_result,
    fake_potential_class,
    score_num_fev,
    score_nodal_points,
):
    """
    Test that a single nodal point is found.
    """
    result = run_node_finder(
        gap_fct=gap_fct,
        initial_mesh_size=INITIAL_MESH_SIZE,
        fake_potential_class=fake_potential_class
    )
    score_num_fev(result)
    score_nodal_points(result, exact_points=[node_position])
    check_result(result)


def test_save(gap_fct, node_position, check_result, fake_potential_class):
    """
    Test saving to a file
    """
    with tempfile.NamedTemporaryFile() as named_file:
        result = run_node_finder(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=INITIAL_MESH_SIZE,
            fake_potential_class=fake_potential_class,
        )
        check_result(result)


def test_restart(gap_fct, node_position, check_result, fake_potential_class):
    """
    Test that the calculation is done when restarting from a finished result.
    """

    def invalid_gap_fct(x):
        raise ValueError

    with tempfile.NamedTemporaryFile() as named_file:
        result = run_node_finder(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=INITIAL_MESH_SIZE,
            fake_potential_class=fake_potential_class
        )
        check_result(result)

        restart_result = run_node_finder(
            gap_fct=invalid_gap_fct,
            save_file=named_file.name,
            load=True,
            load_quiet=False,
            initial_mesh_size=INITIAL_MESH_SIZE,
            fake_potential_class=fake_potential_class
        )
        check_result(restart_result)
