# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests with a nodal line.
"""

import numpy as np
import pytest

from nodefinder.search import run


@pytest.fixture
def nodal_line_properties():
    """
    Fixture which defines the helper functions describing the properties of the
    nodal line.
    """
    radius = 0.2

    def dist_fct(pos):
        dx, dy, dz = (np.array(pos) % 1) - 0.5
        return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2)

    def gap_fct(pos):
        dx, dy, dz = (np.array(pos) % 1) - 0.5
        return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2) * (0.1 + 10 * dx**2)

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
        refinement_stencil="auto",
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
    """
    Fixture which defines the helper functions describing the properties of the
    two 2D nodal lines.
    """

    def gap_fct(pos):
        x, y = pos
        return abs(np.sin(x) + 0.8 * np.cos(y))

    def parametrization(t):
        y = 2 * t % 1
        x = np.arcsin(-0.8 * np.cos(y))  # pylint: disable=assignment-from-no-return,useless-suppression
        if t > 0.5:
            x = np.pi - x
        return np.array([x, y])

    return None, gap_fct, parametrization


def test_nodal_line_2d(nodal_line_2d_properties, score_nodal_line):  # pylint: disable=redefined-outer-name
    """
    Test that two 2D nodal lines are correctly identified.
    """
    dist_fct, gap_fct, parametrization = nodal_line_2d_properties

    result = run(
        gap_fct=gap_fct,
        limits=[(0, 2 * np.pi), (0, 2 * np.pi)],
        gap_threshold=2e-4,
        feature_size=0.05,
        refinement_stencil="auto",
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


@pytest.fixture
def nodal_line_1d_properties():
    """
    Fixture which defines the helper functions describing the properties of the
    two 2D nodal lines.
    """

    def gap_fct(pos):  # pylint: disable=unused-argument
        return 0

    def parametrization(t):
        return np.array([t])

    return None, gap_fct, parametrization


def test_nodal_line_1d(nodal_line_1d_properties, score_nodal_line):  # pylint: disable=redefined-outer-name
    """
    Test that searching a nodal line in 1D works.
    """
    dist_fct, gap_fct, parametrization = nodal_line_1d_properties

    result = run(
        gap_fct=gap_fct,
        limits=[(0, 1)],
        gap_threshold=2e-4,
        feature_size=0.05,
        refinement_stencil="auto",
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


@pytest.fixture
def nodal_line_nonperiodic_properties():  # pylint: disable=invalid-name
    """
    Fixture which defines the helper functions describing the properties of the
    nodal non-periodic line.
    """

    def gap_fct(pos):
        return np.abs(1 - np.max(np.abs(pos)))

    def parametrization(t):
        if t < 0.25:
            return [-1 + 8 * t, -1]
        elif t < 0.5:
            return [1, -1 + 8 * (t - 0.25)]
        elif t < 0.75:
            return [1 - 8 * (t - 0.5), 1]
        else:
            return [-1, 1 - 8 * (t - 0.75)]

    return gap_fct, gap_fct, parametrization


def test_nodal_line_nonperiodic(nodal_line_nonperiodic_properties, score_nodal_line):  # pylint: disable=redefined-outer-name,invalid-name
    """
    Test a nodal line of a non-periodic potential.
    """
    dist_fct, gap_fct, parametrization = nodal_line_nonperiodic_properties

    result = run(
        gap_fct=gap_fct,
        limits=[(-1, 1)] * 2,
        gap_threshold=1e-3,
        feature_size=0.2,
        refinement_stencil="auto",
        initial_mesh_size=3,
        use_fake_potential=True,
        periodic=False,
    )
    score_nodal_line(
        result=result,
        dist_func=dist_fct,
        line_parametrization=parametrization,
        cutoff_accuracy=2e-3,
        cutoff_coverage=0.2,
    )
