"""
Tests with a nodal line.
"""

import numpy as np
import pytest

from nodefinder import run_node_finder
from nodefinder._minimization._fake_potential import FakePotential


@pytest.fixture
def gap_fct_parametrization():
    radius = 0.2

    def gap_fct(pos):
        dx, dy, dz = (np.array(pos) % 1) - 0.5
        return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2)

    def parametrization(t):
        phi = 2 * np.pi * t
        return radius * np.array([np.cos(phi), np.sin(phi), 0]) + 0.5

    return gap_fct, parametrization


def test_nodal_line(gap_fct_parametrization, score_nodal_line):
    """
    Test that a single nodal line is found.
    """
    gap_fct, parametrization = gap_fct_parametrization

    result = run_node_finder(
        gap_fct=gap_fct,
        gap_threshold=2e-4,
        feature_size=1e-2,
        refinement_mesh_size=(3, 3, 3),
        initial_mesh_size=(3, 3, 3),
        fake_potential_class=FakePotential,
    )
    score_nodal_line(
        result=result,
        dist_func=gap_fct,
        line_parametrization=parametrization,
        cutoff_accuracy=1e-3,
        cutoff_coverage=2e-2,
    )
