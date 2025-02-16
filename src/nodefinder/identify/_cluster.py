# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the clustering function used to group nodal points.
"""

import numpy as np
import networkx as nx
from fsc.export import export

_DISTANCE_KEY = "_distance"


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
    list(nx.Graph) :
        A list of connected graphs, each representing one cluster.
    """
    graph = _create_graph(
        positions, feature_size=feature_size, coordinate_system=coordinate_system
    )
    return [
        nx.freeze(graph.subgraph(nodes).copy())
        for nodes in nx.algorithms.connected_components(graph)
    ]


def _create_graph(positions, *, feature_size, coordinate_system):
    """
    Create a graph from the positions, where nodes which are less than the feature
    size apart are connected by an edge, whose 'distance' weight is their
    distance.
    """
    graph = nx.Graph()

    pos_unique = list(set(tuple(pos) for pos in positions))
    graph.add_nodes_from(pos_unique)
    pos_unique_array = np.array(pos_unique)
    for i, (pos, pos_arr) in enumerate(zip(pos_unique[:-1], pos_unique_array)):
        offset = i + 1
        candidates = pos_unique_array[offset:]
        distances = coordinate_system.distance(pos_arr, candidates)
        for idx in np.flatnonzero(distances <= feature_size):
            nbr = pos_unique[idx + offset]
            dist = distances[idx]
            graph.add_edge(pos, nbr, **{_DISTANCE_KEY: dist})
    return graph
