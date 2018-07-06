"""
Tests with a nodal line.
"""

import numpy as np
import pytest

from nodefinder.search import run


@pytest.fixture
def nodal_line_properties():
    radius = 0.2

    def dist_fct(pos):
        dx, dy, dz = (np.array(pos) % 1) - 0.5
        return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2)

    def gap_fct(pos):
        dx, dy, dz = (np.array(pos) % 1) - 0.5
        return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) +
                       dz**2) * (0.1 + 10 * dx**2)

    def parametrization(t):
        phi = 2 * np.pi * t
        return radius * np.array([np.cos(phi), np.sin(phi), 0]) + 0.5

    return dist_fct, gap_fct, parametrization


def test_nodal_line(nodal_line_properties, score_nodal_line):  # pylint: disable=redefined-outer-name
    """
    Test that a single nodal line is found.
    """
    dist_fct, gap_fct, parametrization = nodal_line_properties

    result = run(
        gap_fct=gap_fct,
        gap_threshold=2e-4,
        feature_size=0.05,
        refinement_mesh_size=(3, 3, 3),
        initial_mesh_size=(3, 3, 3),
        use_fake_potential=True,
    )
    score_nodal_line(
        result=result,
        dist_func=dist_fct,
        line_parametrization=parametrization,
        cutoff_accuracy=2e-3,
        cutoff_coverage=0.05,
    )


@pytest.fixture
def nodal_line_2d_properties():
    def gap_fct(pos):
        x, y = pos
        return abs(np.sin(x) + 0.8 * np.cos(y))

    def parametrization(t):
        y = 2 * t % 1
        x = np.arcsin(-0.8 * np.cos(y))
        if t > 0.5:
            x = np.pi - x
        return np.array([x, y])

    return None, gap_fct, parametrization


def test_nodal_line_2d(nodal_line_2d_properties, score_nodal_line):  # pylint: disable=redefined-outer-name
    """
    Test that a 2D nodal line.
    """
    dist_fct, gap_fct, parametrization = nodal_line_2d_properties

    result = run(
        gap_fct=gap_fct,
        limits=[(0, 2 * np.pi), (0, 2 * np.pi)],
        gap_threshold=2e-4,
        feature_size=0.05,
        refinement_mesh_size=3,
        initial_mesh_size=3,
        use_fake_potential=True,
    )
    score_nodal_line(
        result=result,
        dist_func=dist_fct,
        line_parametrization=parametrization,
        cutoff_accuracy=2e-3,
        cutoff_coverage=0.05,
    )
