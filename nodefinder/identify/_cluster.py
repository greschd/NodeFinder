"""
Defines the clustering function used to group nodal points.
"""

import copy
import itertools
from collections import namedtuple

import numpy as np
from fsc.export import export

__all__ = ['Neighbour']

Neighbour = namedtuple('Neighbour', ['pos', 'distance'])


@export
def create_clusters(positions, *, feature_size, coordinate_system):
    """
    Create clusters from the given positions, by joining all positions which are
    less than twice the ``feature_size`` apart.

    Arguments
    ---------
    positions : list(list(float))
        The list of positions to cluster.
    feature_size : float
        Distance between two nodes where they are considered to belong to different clusters.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances between positions.

    Returns
    -------
    list(set) :
        Clusters of positions.
    dict:
        Mapping containing the list of neighbours for each position.
    """
    if not positions:
        return [], {}
    positions_unique = set(tuple(pos) for pos in positions)
    neighbour_mapping = {pos: [] for pos in positions_unique}
    pos_pairs = list(itertools.combinations(positions_unique, r=2))
    pos_pairs_array = np.array(pos_pairs)
    distances = coordinate_system.distance(
        pos_pairs_array[:, 0], pos_pairs_array[:, 1]
    )
    assert len(distances) == len(pos_pairs)
    for idx in np.flatnonzero(distances <= feature_size):
        pos1, pos2 = pos_pairs[idx]
        dist = distances[idx]
        neighbour_mapping[pos1].append(Neighbour(pos=pos2, distance=dist))
        neighbour_mapping[pos2].append(Neighbour(pos=pos1, distance=dist))

    clusters = []
    pos_to_evaluate = copy.copy(positions_unique)

    while pos_to_evaluate:
        new_pos = pos_to_evaluate.pop()
        new_cluster = _create_cluster(
            starting_pos=new_pos, neighbour_mapping=neighbour_mapping
        )
        pos_to_evaluate -= new_cluster
        clusters.append(new_cluster)

    # check consistency
    for cl1, cl2 in itertools.combinations(clusters, r=2):
        assert not cl1.intersection(cl2), "Inconsistent neighbour mapping."

    return clusters, neighbour_mapping


def _create_cluster(starting_pos, neighbour_mapping):
    """Create
    """
    cluster = set([starting_pos])
    get_neighbours_from = set([starting_pos])
    while get_neighbours_from:
        pos = get_neighbours_from.pop()
        neighbours = {n.pos for n in neighbour_mapping[pos]}
        new_positions = neighbours - cluster
        get_neighbours_from.update(new_positions)
        cluster.update(new_positions)
    return cluster
