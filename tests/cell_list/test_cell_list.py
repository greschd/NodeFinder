import numpy as np

from nodefinder._cell_list import CellList


def test_get_frac():
    cl = CellList(cell_size=0.1, limits=[(-0.1, 0.3), (0.4, 0.8)])
    assert np.allclose(cl.get_frac([0.1, 0.6]), [0.5] * 2)


def test_get_index():
    cl = CellList(cell_size=0.099, limits=[(-0.1, 0.3), (0.4, 0.8)])
    assert np.allclose(cl.get_index([0.101, 0.601]), [2, 2])
