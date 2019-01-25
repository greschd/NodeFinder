# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests with a nodal line.
"""

import numpy as np
import pytest

from nodefinder.search import run
from nodefinder.search.refinement_stencil import get_mesh_stencil


@pytest.fixture
def nodal_surface_properties():
    """
    Fixture which defines the helper functions describing the nodal surface.
    """

    def dist_fct(pos):
        _, _, dz = (np.array(pos) % 1) - 0.5
        return abs(dz)

    def gap_fct(pos):
        dx, _, dz = (np.array(pos) % 1) - 0.5
        return dz**2 * (0.1 + 10 * dx**2)

    def parametrization(s, t):
        return [s, t, 0.5]

    return dist_fct, gap_fct, parametrization


def test_nodal_surface(nodal_surface_properties, score_nodal_surface):  # pylint: disable=redefined-outer-name
    """
    Test that a nodal surface is found.
    """
    dist_fct, gap_fct, parametrization = nodal_surface_properties

    result = run(
        gap_fct=gap_fct,
        gap_threshold=1e-4,
        feature_size=1e-1,
        refinement_stencil=get_mesh_stencil(mesh_size=(2, 2, 2)),
        initial_mesh_size=(3, 3, 3),
        use_fake_potential=False,
    )
    score_nodal_surface(
        result=result,
        dist_func=dist_fct,
        surface_parametrization=parametrization,
        cutoff_accuracy=2e-3,
        cutoff_coverage=1e-1,
    )
