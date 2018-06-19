import numpy as np
from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5

from ._cell_list import CellList


@export
@subscribe_hdf5('nodefinder.result')
class Result(HDF5Enabled):
    def __init__(
        self,
        *,
        coordinate_system,
        minimization_results=(),
        gap_threshold,
        dist_cutoff
    ):
        self.coordinate_system = coordinate_system
        self.gap_threshold = gap_threshold
        self.dist_cutoff = dist_cutoff

        num_cells = np.maximum(
            1,
            np.array(
                self.coordinate_system.size / self.dist_cutoff, dtype=int
            )
        )
        self.nodes = CellList(num_cells=num_cells)
        self.rejected_results = []
        for res in minimization_results:
            self.add_result(res)

    def add_result(self, res):
        res.pos = self.coordinate_system.normalize_position(res.pos)
        if not res.success or res.value > self.gap_threshold:
            self.rejected_results.append(res)
        else:
            self.nodes.add_point(self.coordinate_system.get_frac(res.pos), res)

    @property
    def results(self):
        return self.nodes.values() + self.rejected_results

    def get_node_neighbours(self, pos):
        candidates = self.nodes.get_neighbour_values(
            frac=self.coordinate_system.get_frac(pos)
        )
        return [
            c for c in candidates
            if self.coordinate_system.distance(pos, c.pos) < self.dist_cutoff
        ]

    def to_hdf5(self, hdf5_handle):
        coord_group = hdf5_handle.create_group('coordinate_system')
        to_hdf5(self.coordinate_system, coord_group)
        hdf5_handle['gap_threshold'] = self.gap_threshold
        hdf5_handle['dist_cutoff'] = self.dist_cutoff
        res_group = hdf5_handle.create_group('minimization_results')
        to_hdf5(self.results, res_group)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            coordinate_system=from_hdf5(hdf5_handle['coordinate_system']),
            gap_threshold=hdf5_handle['gap_threshold'].value,
            dist_cutoff=hdf5_handle['dist_cutoff'].value,
            minimization_results=from_hdf5(
                hdf5_handle['minimization_results']
            ),
        )
