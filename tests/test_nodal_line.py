"""
Tests with a nodal line.
"""

import numpy as np
import pytest

from nodefinder import run_node_finder

@pytest.fixture
def gap_fct():
    radius = 0.1
    def inner(pos):
        x, y, z = pos
        return np.sqrt(np.abs(x**2 + y**2 - radius**2) + z**2)
    return inner

def test_nodal_line(gap_fct):
    """
    Test that a single nodal line is found.
    """
    radius = 0.1
    xtol = 1e-6

    result = run_node_finder(
        gap_fct=gap_fct,
        feature_size=1e-2,
        refinement_box_size=5e-2,
        num_minimize_parallel=100
    )

    all_nodes = result.nodes.values()
    # assert len(all_nodes) > 32
    for node in all_nodes:
        assert np.isclose(node.value, 0, atol=1e-6)
        assert np.isclose(
            result.coordinate_system.distance(node.pos, (0, 0, 0)), radius, atol=2e-6
        )
