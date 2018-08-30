"""
Defines the functions used to evaluate the shape of a given cluster of points.
"""

import copy
import operator
import warnings
import itertools
from collections import Counter, namedtuple

import numpy as np
import networkx as nx

from fsc.export import export

from ..search._controller import _DIST_CUTOFF_FACTOR
from .result import NodalLine, NodalPoint
from ._logging import IDENTIFY_LOGGER

_DISTANCE_KEY = '_distance'
_WEIGHT_KEY = '_weight'


@export
def evaluate_cluster(
    positions, dim, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the shape of a cluster with the given positions.

    Arguments
    ---------
    positions : set(tuple(float))
        Positions of the points in the cluster.
    dim : int
        Dimension of the cluster.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
    neighbour_mapping : dict
        Mapping containing a list of neighbours for each position.
    feature_size : float
        Distance between two nodal points at which they are considered distinct.

    Returns
    -------
    :obj:`None` or NodalPoint or NodalLine :
        The shape of the given positions. Returns ``None`` if the shape could
        not be determined.
    """
    if dim == 0:
        return _evaluate_point(
            positions=positions, coordinate_system=coordinate_system
        )
    elif dim == 1:
        try:
            return _evaluate_line(
                positions=positions,
                coordinate_system=coordinate_system,
                neighbour_mapping={p: neighbour_mapping[p]
                                   for p in positions},
                feature_size=feature_size
            )
        except (IndexError, ValueError) as exc:
            warnings.warn('Could not identify line: {}'.format(exc))
    else:
        return None


def _evaluate_point(positions, coordinate_system):
    return NodalPoint(
        position=coordinate_system.
        average([np.array(pos) for pos in positions])
    )


def _evaluate_line(
    positions, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the positions of a closed line.
    """
    graph = nx.Graph()
    graph.add_nodes_from(positions)
    graph_reduced_neighbours = copy.deepcopy(graph)
    for node, neighbours in neighbour_mapping.items():
        for nbr in neighbours:
            if nbr not in graph.nodes:
                edge = (node, nbr.pos)
                graph.add_edge(
                    *edge, **{
                        _DISTANCE_KEY: nbr.distance,
                        _WEIGHT_KEY: nbr.distance**4
                    }
                )
                if nbr.distance < feature_size / _DIST_CUTOFF_FACTOR:
                    graph_reduced_neighbours.add_edge(*edge)

    dominating_set = nx.algorithms.dominating_set(graph_reduced_neighbours)
    subgraph = graph.subgraph(dominating_set).copy()

    _patch_all_subgraph_holes(
        subgraph=subgraph,
        graph=graph,
        coordinate_system=coordinate_system,
        feature_size=feature_size
    )
    _remove_duplicate_paths(subgraph)

    degree_counter = Counter([val for pos, val in subgraph.degree])
    degree_counter.pop(2, None)
    return NodalLine(graph=subgraph, degree_count=degree_counter)


def _patch_all_subgraph_holes(
    *, subgraph, graph, coordinate_system, feature_size
):
    """
    Check for 'holes' where the subgraph is disconnected while the original graph is not, and patch them by adding the shortest path on the full graph between the points in the subgraph.
    """
    pair_to_patch = namedtuple('pair_to_patch', ['edge', 'dist_original'])
    to_patch = []
    for pos1, pos2 in itertools.combinations(subgraph.nodes, r=2):
        # The minimum distance is used here only to improve performance,
        # because it is much quicker to calculate than the shortest path.
        # By using the distance in the coordinate system as a lower limit for
        # 'dist_original', we can abort early in many cases.
        dist_minimal = coordinate_system.distance(
            np.array(pos1), np.array(pos2)
        )
        if dist_minimal < 2 * feature_size:
            dist_reduced = nx.algorithms.shortest_path_length(
                subgraph, pos1, pos2, weight=_DISTANCE_KEY
            )
            if dist_reduced > max(feature_size, 2 * dist_minimal):
                dist_original = nx.algorithms.shortest_path_length(
                    graph, pos1, pos2, weight=_DISTANCE_KEY
                )
                if dist_reduced > 2 * dist_original and dist_original < 2 * feature_size:
                    to_patch.append(
                        pair_to_patch(
                            edge=(pos1, pos2), dist_original=dist_original
                        )
                    )

    to_patch = sorted(to_patch, key=operator.attrgetter('dist_original'))
    for edge, dist_original in to_patch:
        # might have changed since the subgraph is being patched
        dist_reduced = nx.algorithms.shortest_path_length(
            subgraph, *edge, weight=_DISTANCE_KEY
        )
        if dist_reduced > 2 * dist_original:
            IDENTIFY_LOGGER.debug(
                'Patching hole {} in sub-graph.'.format(edge)
            )
            start, end = edge
            _patch_subgraph_hole(
                subgraph=subgraph, graph=graph, start=start, end=end
            )


def _patch_subgraph_hole(*, subgraph, graph, start, end):
    """
    Patch a single hole in the subgraph between the given start and end points.
    """
    shortest_path = nx.algorithms.shortest_path(
        graph, start, end, weight=_DISTANCE_KEY
    )
    for node in shortest_path[1:-1]:
        subgraph.add_node(node)
    for edge in zip(shortest_path[:-1], shortest_path[1:]):
        subgraph.add_edge(*edge, **graph.edges[edge])


def _remove_duplicate_paths(subgraph):
    """
    Remove all "duplicate" edges from the graph where there is another path with lower total weight connecting the two points.
    """
    for *edge, _ in sorted(
        subgraph.edges(data=_WEIGHT_KEY), key=lambda e: -e[2]
    ):
        shortest_path = nx.algorithms.shortest_path(
            subgraph, *edge, weight=_WEIGHT_KEY
        )
        if len(shortest_path) > 2:
            subgraph.remove_edge(*edge)
