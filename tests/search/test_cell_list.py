# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the CellList container.
"""

from nodefinder.search.result._cell_list import CellList


def test_get_index():
    """
    Test that the index is correctly computed.
    """
    cell_list = CellList(num_cells=(3, 5, 2), periodic=True)
    assert cell_list.get_index([0.9, 0.19, 0.4]) == (2, 0, 0)


def test_get_index_nonperiodic():
    """
    Test that the index is correctly computed.
    """
    cell_list = CellList(num_cells=(3, 5, 2), periodic=False)
    assert cell_list.get_index([-1, 2, -1]) == (0, 6, 0)
    assert cell_list.get_index([0.9, 0.19, 0.4]) == (3, 1, 1)
