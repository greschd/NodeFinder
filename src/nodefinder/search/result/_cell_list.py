# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the cell list class, used for quicker lookup of the neighbours of a
given position.
"""

import itertools

import numpy as np
from fsc.export import export


@export
class CellList:
    """
    Cell list container for the NodalPoint objects.
    """

    def __init__(self, num_cells, periodic):
        self.periodic = periodic
        self.num_cells = np.array(num_cells, dtype=int)
        self._total_num_cells = np.copy(self.num_cells)
        if not self.periodic:
            self._total_num_cells += 2  # add 'boundary' boxes for outside points.
        assert np.all(self.num_cells > 0)
        self._cells = np.empty(shape=self._total_num_cells, dtype=object)
        filler = np.frompyfunc(lambda x: list(), 1, 1)
        filler(self._cells, self._cells)
        self._values_flat = []

        self._neighbour_offset = np.array(
            list(itertools.product([-1, 0, 1], repeat=len(self.num_cells)))
        )
        self._neighbour_indices = dict()

    def _get_neighbour_indices(self, idx):
        """
        Get the indices of neighbouring cells for a given index.
        """
        try:
            return self._neighbour_indices[idx]
        except KeyError:
            res = self._calculate_neighbour_indices(idx)
            self._neighbour_indices[idx] = res
            return res

    def _calculate_neighbour_indices(self, idx):  # pylint: disable=missing-function-docstring
        indices = idx + self._neighbour_offset

        if self.periodic:
            return [tuple(i % self._total_num_cells) for i in indices]
        return [
            tuple(i)
            for i in indices
            if np.all(i >= 0) and np.all(i < self._total_num_cells)
        ]

    def add_point(self, frac, value):
        idx = self.get_index(frac)
        self._cells[idx].append(value)
        self._values_flat.append(value)

    def get_index(self, frac):  # pylint: disable=missing-function-docstring
        vals = np.array(frac * self.num_cells, dtype=int)
        if not self.periodic:
            vals += 1
            vals = np.maximum(0, np.minimum(vals, self._total_num_cells - 1))  # pylint: disable=assignment-from-no-return,useless-suppression
        return tuple(vals)

    def values(self):
        return self._values_flat

    def __len__(self):
        return len(self._values_flat)

    def __iter__(self):
        return iter(self._values_flat)

    def __getitem__(self, key):
        return self._values_flat[key]

    def get_neighbour_values(self, frac):
        idx = self.get_index(frac)
        for cell_idx in self._get_neighbour_indices(idx):
            yield from self._cells[cell_idx]
