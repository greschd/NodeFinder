"""
Tests with a single nodal point.
"""

import numpy as np

from nodefinder import NodeFinder, periodic_distance


def test_nodal_line():
    """
    Test that a single nodal point is found.
    """
    radius = 0.1
    xtol = 1e-6

    def gap_fct(x):
        kx, ky, kz = x
        return np.abs(kx**2 + ky**2 - radius**2) + kz**2

    node_finder = NodeFinder(
        gap_fct=gap_fct,
        feature_size=1e-2,
        refinement_box_size=5e-2,
        xtol=xtol
    )
    node_finder.run()
    all_nodes = node_finder.nodal_points
    assert len(all_nodes) > 32
    for node in all_nodes:
        assert np.isclose(node.gap, 0, atol=1e-6)
        assert np.isclose(
            periodic_distance(node.k, (0, 0, 0)), radius, atol=2 * xtol
        )
