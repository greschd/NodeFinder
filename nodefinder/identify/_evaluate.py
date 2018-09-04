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
    positions,
    dim,
    coordinate_system,
    neighbour_mapping,
    feature_size,
    evaluate_line_method='shortest_path'
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
                feature_size=feature_size,
                method=evaluate_line_method,
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
    positions,
    coordinate_system,
    neighbour_mapping,
    feature_size,
    method='shortest_path'
):
    """
    Evaluate the positions of a nodal line.
    """
    if method == 'shortest_path':
        return _evaluate_line_shortest_path(
            positions=positions,
            coordinate_system=coordinate_system,
            neighbour_mapping=neighbour_mapping,
            feature_size=feature_size
        )
    elif method == 'dominating_set':
        return _evaluate_line_dominating_set(
            positions=positions,
            coordinate_system=coordinate_system,
            neighbour_mapping=neighbour_mapping,
            feature_size=feature_size
        )
    else:
        raise ValueError('Invalid value for \'method\': {}'.format(method))


def _evaluate_line_shortest_path(
    positions, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the positions of a nodal line using the 'shortest path' method.
    """
    graph = nx.Graph()
    graph.add_nodes_from(positions)
    for node, neighbours in neighbour_mapping.items():
        if node in positions:
            distance_func = lambda x: x + x**2
            graph.add_weighted_edges_from([(
                node, nbr.pos,
                feature_size * distance_func(nbr.distance / feature_size)
            ) for nbr in neighbours],
                                          weight=_WEIGHT_KEY)
    graph_unmodified = graph.copy()

    candidate_positions = set(graph.nodes)

    first_pos = _get_first_position(
        candidate_positions=candidate_positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size
    )
    IDENTIFY_LOGGER.debug('Starting from position {}.'.format(first_pos))
    start_neighbours = set(graph.neighbors(first_pos)) | {first_pos}
    candidate_positions -= start_neighbours

    result_graph = nx.Graph()
    result_graph.add_node(first_pos)

    high_degree_nodes = set()
    # num_pos = 0
    for num_pos in itertools.count():
        # while candidate_positions:
        if not candidate_positions:
            break
        # prioritize "crossings"
        _update_high_degree_nodes(graph, result_graph, high_degree_nodes)

        tmp_graph = graph.copy()
        end_pos = None
        min_dist = 0
        # TODO: Calculate all distances only once.
        for pos in candidate_positions:
            dist = min(
                coordinate_system.distance(np.array(pos), np.array(node))
                for node in result_graph.nodes
            )
            if dist > min_dist:
                min_dist = dist
                end_pos = pos

        if min_dist < feature_size:
            break
        elif min_dist < 3 * feature_size:
            single_path_only = True
            num_pos += 1
        else:
            single_path_only = False
            num_pos += 1

        end_neighbours = set(graph.neighbors(end_pos)) | {end_pos}
        candidate_positions -= end_neighbours
        start_end_neighbours = start_neighbours | end_neighbours

        for num_paths in itertools.count():
            if single_path_only and num_paths > 0:
                break
            try:
                path = nx.algorithms.shortest_path(
                    tmp_graph,
                    source=first_pos,
                    target=end_pos,
                    weight=_WEIGHT_KEY,
                )
            except nx.NetworkXNoPath:
                break
            edges = list(zip(path, path[1:]))
            result_graph.add_edges_from(edges)
            for edge in edges:
                graph.edges[edge][_WEIGHT_KEY] = 0
                tmp_graph.edges[edge][_WEIGHT_KEY] = 0
            new_neighbours = set()
            for node in path:
                new_neighbours.update(graph.neighbors(node))
                new_neighbours.add(node)
            candidate_positions -= new_neighbours

            nodes_to_remove = new_neighbours - start_end_neighbours
            tmp_graph.remove_nodes_from(nodes_to_remove)
        if num_paths > 0:  # pylint: disable=undefined-loop-variable
            IDENTIFY_LOGGER.debug(
                'Found {} path(s) to point {}.'.format(num_paths, end_pos)  # pylint: disable=undefined-loop-variable
            )
        else:
            IDENTIFY_LOGGER.warning(
                'No paths to point {} found.'.format(end_pos)
            )
    if num_pos > 0:
        IDENTIFY_LOGGER.debug(
            'Calcualted paths to {} positions.'.format(num_pos)
        )
    else:
        IDENTIFY_LOGGER.warning(
            'Could not find any positions with sufficient spacing using the \'shortest_path\' method. Using \'dominating_set\' method instead.'
        )
        return _evaluate_line_dominating_set(
            positions=positions,
            coordinate_system=coordinate_system,
            neighbour_mapping=neighbour_mapping,
            feature_size=feature_size
        )

    for edge in result_graph.edges:
        result_graph.edges[edge][_WEIGHT_KEY
                                 ] = graph_unmodified.edges[edge][_WEIGHT_KEY]

    unusual_nodes = [
        node for node, degree in result_graph.degree if degree != 2
    ]
    _patch_all_subgraph_holes(
        subgraph=result_graph,
        graph=graph_unmodified,
        coordinate_system=coordinate_system,
        feature_size=feature_size,
        candidates=unusual_nodes,
        weight=_WEIGHT_KEY
    )

    return NodalLine(
        graph=result_graph, degree_count=_create_degree_count(result_graph)
    )


def _get_first_position(candidate_positions, coordinate_system, feature_size):
    """
    Get the first position in such a way that it there is a second position that
    is far away.
    """
    first_pos = None
    max_dist = 0
    for pos1, pos2 in itertools.combinations(candidate_positions, r=2):
        dist = coordinate_system.distance(np.array(pos1), np.array(pos2))
        if dist > max_dist:
            first_pos = pos1
            max_dist = dist
        if max_dist > 3 * feature_size:
            break
    return first_pos


def _update_high_degree_nodes(graph, result_graph, high_degree_nodes):
    """
    Update the high degree nodes, and adjust the weight of the edges for new
    high degree nodes.
    """
    for node, deg in result_graph.degree:
        if deg > 2 and node not in high_degree_nodes:
            high_degree_nodes.add(node)
            for nbr in graph.neighbors(node):
                graph.edges[(node, nbr)][_WEIGHT_KEY] *= 0.5


def _create_degree_count(graph):
    degree_counter = Counter([val for pos, val in graph.degree])
    degree_counter.pop(2, None)
    return degree_counter


def _evaluate_line_dominating_set(
    positions, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the positions of a nodal line using the 'dominating set' method.
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
        feature_size=feature_size,
    )
    _remove_duplicate_paths(subgraph)

    return NodalLine(
        graph=subgraph, degree_count=_create_degree_count(subgraph)
    )


def _patch_all_subgraph_holes(
    *,
    subgraph,
    graph,
    coordinate_system,
    feature_size,
    candidates=None,
    weight=_DISTANCE_KEY
):
    """
    Check for 'holes' where the subgraph is disconnected while the original graph is not, and patch them by adding the shortest path on the full graph between the points in the subgraph.
    """
    pair_to_patch = namedtuple('pair_to_patch', ['edge', 'dist_original'])
    to_patch = []

    patch_cutoff = 1.001
    if candidates is None:
        candidates = subgraph.nodes
    for pos1, pos2 in itertools.combinations(candidates, r=2):
        # The minimum distance is used here only to improve performance,
        # because it is much quicker to calculate than the shortest path.
        # By using the distance in the coordinate system as a lower limit for
        # 'dist_original', we can abort early in many cases.
        dist_minimal = coordinate_system.distance(
            np.array(pos1), np.array(pos2)
        )
        if dist_minimal < 2 * feature_size:
            dist_reduced = nx.algorithms.shortest_path_length(
                subgraph, pos1, pos2, weight=weight
            )
            if dist_reduced > max(feature_size, patch_cutoff * dist_minimal):
                dist_original = nx.algorithms.shortest_path_length(
                    graph, pos1, pos2, weight=weight
                )
                if dist_reduced > patch_cutoff * dist_original and dist_original < 2 * feature_size:
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
