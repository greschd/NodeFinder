"""
Tests for the CellList container.
"""

from nodefinder.search.result._cell_list import CellList


def test_get_index():
    """
    Test that the index is correctly computed.
    """
    cell_list = CellList(num_cells=(3, 5, 2))
    assert cell_list.get_index([0.9, 0.19, 0.4]) == (2, 0, 0)
