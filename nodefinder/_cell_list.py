import itertools
import contextlib

import numpy as np
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5

from ._utils import periodic_distance


@subscribe_hdf5('nodefinder.cell_list')
class CellList(HDF5Enabled):
    """
    Cell list container for the NodalPoint objects.
    """

    def __init__(self, cell_size, points=()):
        self._num_cells = max(1, int(1 / cell_size))
        self._cell_size = 1 / self._num_cells
        self._cells = dict()
        for p in points:
            self.add_point(p)

    def get_index(self, k):
        return tuple(int(ki * self._num_cells) for ki in k)

    def add_point(self, point):
        idx = self.get_index(point.k)
        try:
            self._cells[idx].append(point)
        except KeyError:
            self._cells[idx] = [point]

    @property
    def points(self):
        return sum(self._cells.values(), [])

    def minimum_distance(self, k, cutoff):
        """
        Returns the distance from ``k`` to the nearest point in the container which is closer than the ``cutoff``. Returns ``None`` if no point is within ``cutoff`` of ``k``.
        """
        with contextlib.suppress(ValueError):
            dist = min(
                periodic_distance(k, point.k)
                for point in self._get_neighbours(k, min_cutoff=cutoff)
            )
            if dist <= cutoff:
                return dist

    def _get_neighbours(self, k, min_cutoff):
        idx = self.get_index(k)
        cell_indices = self._get_neighbour_cell_indices(
            k, min_cutoff=min_cutoff
        )
        return sum(
            (self._cells.get(cell_idx, []) for cell_idx in cell_indices), []
        )

    def _get_neighbour_cell_indices(self, k, min_cutoff):
        idx = self.get_index(k)
        cell_distance = int(np.ceil(min_cutoff / self._cell_size))
        ranges = [[
            x % self._num_cells
            for x in range(i - cell_distance, i + cell_distance + 1)
        ] for i in idx]
        return list(itertools.product(ranges))

    def to_hdf5(self, hdf5_handle):
        hdf5_handle['num_cells'] = self._num_cells
        points_group = hdf5_handle.create_group('points')
        to_hdf5(self.points, points_group)

    @classmethod
    def from_hdf5(cls, hdf5_handle, cell_size=None):
        if cell_size is None:
            num_cells = hdf5_handle['num_cells'].value
            cell_size = 1 / (
                num_cells + 0.5
            )  # make sure that the resulting num_cells will match
        points = from_hdf5(hdf5_handle['points'])
        return cls(cell_size=cell_size, points=points)
