# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the top-level result class for the search step.
"""

import numpy as np
from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5

from ._cell_list import CellList


@export
@subscribe_hdf5(
    "nodefinder.search_result_container", extra_tags=["nodefinder.result_container"]
)
class SearchResultContainer(SimpleHDF5Mapping):
    """
    Container for the results of a search run.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        Coordinate system used.
    nodes : list(MinimizationResult)
        Minimization results which fulfill the gap threshold criterion.
    gap_threshold : float
        Threshold for results to be considered a node.
    dist_cutoff : float
        Cutoff distance for searching neighbouring nodes.

    """

    HDF5_ATTRIBUTES = [
        "coordinate_system",
        "minimization_results",
        "dist_cutoff",
        "gap_threshold",
    ]
    HDF5_OPTIONAL = ["refined_results"]

    def __init__(
        self,
        *,
        coordinate_system,
        minimization_results=(),
        gap_threshold,
        dist_cutoff,
        refined_results=(),
    ):
        self.coordinate_system = coordinate_system
        self.gap_threshold = gap_threshold
        self.dist_cutoff = dist_cutoff

        if dist_cutoff == 0:
            num_cells = np.full_like(self.coordinate_system.size, 100)
        else:
            num_cells = np.minimum(  # pylint: disable=assignment-from-no-return,useless-suppression
                100,
                np.maximum(
                    1,
                    np.array(self.coordinate_system.size / self.dist_cutoff, dtype=int),
                ),
            )
        self.nodes = CellList(
            num_cells=num_cells, periodic=self.coordinate_system.periodic
        )
        self.rejected_results = []
        for res in minimization_results:
            self.add_result(res)
        self.refined_results = CellList(
            num_cells=num_cells, periodic=self.coordinate_system.periodic
        )
        for res in refined_results:
            self.set_refined(res)
        self.needs_saving = True

    def __repr__(self):
        return "SearchResultContainer(coordinate_system={0.coordinate_system}, minimization_results=<{1} values>, gap_threshold={0.gap_threshold!r}, dist_cutoff={0.dist_cutoff!r})".format(
            self, len(self.minimization_results)
        )

    def add_result(self, res):
        """
        Add a minimization result to the container.

        Arguments
        ---------
        res : MinimizationResult
            Minimization result to add.
        """
        res.pos = self.coordinate_system.normalize_position(res.pos)
        if not res.success or res.value > self.gap_threshold:  # pylint: disable=no-else-return
            self.rejected_results.append(res)
            return False
        else:
            self.nodes.add_point(self.coordinate_system.get_frac(res.pos), res)
            return True
        self.needs_saving = True

    def set_refined(self, pos):
        """
        Set a position to be refined.

        Arguments
        ---------
        pos : np.array
            The position from where refinement started.
        """
        self.refined_results.add_point(self.coordinate_system.get_frac(pos), pos)
        self.needs_saving = True

    @property
    def minimization_results(self):
        """
        list(MinimizationResult):
            All minimization results, including rejected points.
        """
        return self.nodes.values() + self.rejected_results

    def _get_neighbour_iterator(self, pos):
        candidates = self.nodes.get_neighbour_values(
            frac=self.coordinate_system.get_frac(pos)
        )
        return (c for c in candidates if np.any(c.pos != pos))

    def get_neighbour_distance_iterator(self, pos):
        """
        Returns an iterator over the distance to neighbouring nodes from a given
        position. Only neighbours within ``dist_cutoff`` are taken into account.

        Arguments
        ---------
        pos : numpy.ndarray
            Position for which to calculate the distances.
        """
        candidates = self._get_neighbour_iterator(pos)
        return (self.coordinate_system.distance(pos, c.pos) for c in candidates)

    def get_refined_neighbour_distance_iterator(self, pos):  # pylint: disable=invalid-name
        """
        Returns an iterator over the distance to neighboring nodes which have
        been used as a starting point in a refinement procedure.

        Arguments
        ---------
        pos : numpy.ndarray
            Position for which to calculate the distances.
        """
        candidates = self.refined_results.get_neighbour_values(
            frac=self.coordinate_system.get_frac(pos)
        )
        return (self.coordinate_system.distance(pos, c) for c in candidates)

    def get_all_neighbour_distances(self, pos):
        """
        Calculate the distances to neighbouring nodes from a given position.
        Only neighbours within ``dist_cutoff`` are taken into account.

        Arguments
        ---------
        pos : numpy.ndarray
            Position for which to calculate the distances.
        """
        candidates = self._get_neighbour_iterator(pos)
        positions = np.array([c.pos for c in candidates])

        if positions.size == 0:
            return []
        return self.coordinate_system.distance(pos, positions)
