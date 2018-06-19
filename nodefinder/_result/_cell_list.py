import itertools
from functools import lru_cache

import numpy as np
from fsc.export import export


@export
class CellList:
    """
    Cell list container for the NodalPoint objects.
    """
    def __init__(self, num_cells):
        self.num_cells = np.array(num_cells, dtype=int)
        assert np.all(num_cells > 0)
        self._cells = np.empty(shape=self.num_cells, dtype=object)
        for i, _ in enumerate(self._cells.flat):
            self._cells.flat[i] = []

    def add_point(self, frac, value):
        idx = self._get_index(frac)
        self._cells[idx].append(value)

    def get_index(self, frac):
        return np.array(frac * self.num_cells, dtype=int)

    def values(self):
        return sum(self._cells.values(), [])

    def __iter__(self):
        return iter(self.values())

    def get_neighbour_values(self, frac, periodic=False):
        idx = self.get_index(frac)
        return sum((
            self._cells[cell_idx] for cell_idx in
            self.get_neighbour_indices(idx=idx, periodic=periodic)
        ), [])

    def get_neighbour_indices(self, idx, periodic=False):
        indices = [offset + idx for offset in self._get_offsets()]
        if periodic:
            return list(set(i % self._num_cells for i in indices))
        else:
            return [
                i for i in indices if np.all(0 <= i) and np.all(i < self._num_cells)
            ]

    @lru_cache
    def _get_offsets(self):
        return [
            np.array(o)
            for o in itertools.product([-1, 0, 1], repeat=len(self.num_cells))
        ]
