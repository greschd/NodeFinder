from nodefinder._result._cell_list import CellList


def test_get_index():
    cl = CellList(num_cells=(3, 5, 2))
    idx = cl.get_index([0.9, 0.19, 0.4])
    assert idx == (2, 0, 0)
