# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Implements line evaluation with the dominating set method.
"""

import operator
import itertools
from collections import namedtuple

import numpy as np
import networkx as nx

from ....search._controller import _DIST_CUTOFF_FACTOR
from ..._cluster import _DISTANCE_KEY
from ..._logging import IDENTIFY_LOGGER

_WEIGHT_KEY = "_weight"


def _evaluate_line_dominating_set(*, graph, coordinate_system, feature_size):
    """
    Evaluate the positions of a nodal line using the 'dominating set' method.
    """
    graph_copy = graph.copy()
    graph_reduced_neighbours = nx.Graph()
    graph_reduced_neighbours.add_nodes_from(graph_copy.nodes)

    for edge in graph_copy.edges:
        edge_attrs = graph_copy.edges[edge]
        distance = edge_attrs[_DISTANCE_KEY]
        edge_attrs[_WEIGHT_KEY] = distance**4
        if distance < feature_size / _DIST_CUTOFF_FACTOR:
            graph_reduced_neighbours.add_edge(*edge)

    dominating_set = nx.algorithms.dominating_set(graph_reduced_neighbours)
    subgraph = graph.subgraph(dominating_set).copy()

    _patch_all_subgraph_holes(
        subgraph=subgraph,
        graph=graph_copy,
        coordinate_system=coordinate_system,
        feature_size=feature_size,
    )
    _remove_duplicate_paths(subgraph)

    return subgraph


def _patch_all_subgraph_holes(
    *,
    subgraph,
    graph,
    coordinate_system,
    feature_size,
    candidates=None,
    weight=_DISTANCE_KEY,
):
    """
    Check for 'holes' where the subgraph is disconnected while the original graph is not, and patch them by adding the shortest path on the full graph between the points in the subgraph.
    """
    pair_to_patch = namedtuple("pair_to_patch", ["edge", "dist_original"])
    to_patch = []

    patch_cutoff = 1.001
    if candidates is None:
        candidates = subgraph.nodes
    for pos1, pos2 in itertools.combinations(candidates, r=2):
        # The minimum distance is used here only to improve performance,
        # because it is much quicker to calculate than the shortest path.
        # By using the distance in the coordinate system as a lower limit for
        # 'dist_original', we can abort early in many cases.
        dist_minimal = coordinate_system.distance(np.array(pos1), np.array(pos2))
        if dist_minimal < 2 * feature_size:
            dist_reduced = nx.algorithms.shortest_path_length(
                subgraph, pos1, pos2, weight=weight
            )
            if dist_reduced > max(feature_size, patch_cutoff * dist_minimal):
                dist_original = nx.algorithms.shortest_path_length(
                    graph, pos1, pos2, weight=weight
                )
                if (
                    dist_reduced > patch_cutoff * dist_original
                    and dist_original < 2 * feature_size
                ):
                    to_patch.append(
                        pair_to_patch(edge=(pos1, pos2), dist_original=dist_original)
                    )

    to_patch = sorted(to_patch, key=operator.attrgetter("dist_original"))
    for edge, dist_original in to_patch:
        # might have changed since the subgraph is being patched
        dist_reduced = nx.algorithms.shortest_path_length(
            subgraph, *edge, weight=_DISTANCE_KEY
        )
        if dist_reduced > 2 * dist_original:
            IDENTIFY_LOGGER.debug("Patching hole {} in sub-graph.".format(edge))
            start, end = edge
            _patch_subgraph_hole(subgraph=subgraph, graph=graph, start=start, end=end)


def _patch_subgraph_hole(*, subgraph, graph, start, end):
    """
    Patch a single hole in the subgraph between the given start and end points.
    """
    shortest_path = nx.algorithms.shortest_path(graph, start, end, weight=_DISTANCE_KEY)
    for node in shortest_path[1:-1]:
        subgraph.add_node(node)
    for edge in zip(shortest_path[:-1], shortest_path[1:]):
        subgraph.add_edge(*edge, **graph.edges[edge])


def _remove_duplicate_paths(subgraph):
    """
    Remove all "duplicate" edges from the graph where there is another path with lower total weight connecting the two points.
    """
    for *edge, _ in sorted(subgraph.edges(data=_WEIGHT_KEY), key=lambda e: -e[2]):
        shortest_path = nx.algorithms.shortest_path(subgraph, *edge, weight=_WEIGHT_KEY)
        if len(shortest_path) > 2:
            subgraph.remove_edge(*edge)
