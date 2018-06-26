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
        assert np.all(self.num_cells > 0)
        self._cells = np.empty(shape=self.num_cells, dtype=object)
        for i, _ in enumerate(self._cells.flat):
            self._cells.flat[i] = []
        self._values_flat = []

    def add_point(self, frac, value):
        idx = self.get_index(frac)
        self._cells[idx].append(value)
        self._values_flat.append(value)

    def get_index(self, frac):
        return tuple(np.array(frac * self.num_cells, dtype=int))

    def values(self):
        return self._values_flat

    def __len__(self):
        return len(self._values_flat)

    def __iter__(self):
        return iter(self._values_flat)

    def __getitem__(self, key):
        return self._values_flat[key]

    def get_neighbour_values(self, frac, periodic=True):
        idx = self.get_index(frac)
        for cell_idx in self.get_neighbour_indices(idx=idx, periodic=periodic):
            yield from self._cells[cell_idx]

    @lru_cache(maxsize=None)
    def get_neighbour_indices(self, idx, periodic=True):
        indices = [offset + idx for offset in self._get_offsets()]
        if periodic:
            return [tuple(i % self.num_cells) for i in indices]
        return [
            tuple(i) for i in indices
            if np.all(i >= 0) and np.all(i < self.num_cells)
        ]

    @lru_cache(maxsize=None)
    def _get_offsets(self):
        return [
            np.array(o)
            for o in itertools.product([-1, 0, 1], repeat=len(self.num_cells))
        ]
