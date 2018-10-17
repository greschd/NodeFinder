"""
Defines the functions used to evaluate the shape of a given cluster of points.
"""

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
from ._cluster import _DISTANCE_KEY

_WEIGHT_KEY = '_weight'
_MAX_NUM_PATHS = 20


@export
def evaluate_cluster(
    graph,
    dim,
    coordinate_system,
    feature_size,
    evaluate_line_method='shortest_path'
):
    """
    Evaluate the shape of a cluster with the given positions.

    Arguments
    ---------
    graph : nx.Graph
        Graph describing the cluster.
    dim : int
        Dimension of the cluster.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
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
            positions=list(graph.nodes), coordinate_system=coordinate_system
        )
    elif dim == 1:
        try:
            return _evaluate_line(
                graph=graph,
                coordinate_system=coordinate_system,
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
    graph, coordinate_system, feature_size, method='shortest_path'
):
    """
    Evaluate the positions of a nodal line.
    """
    if method == 'shortest_path':
        return _evaluate_line_shortest_path(
            graph=graph,
            coordinate_system=coordinate_system,
            feature_size=feature_size
        )
    elif method == 'dominating_set':
        return _evaluate_line_dominating_set(
            graph=graph,
            coordinate_system=coordinate_system,
            feature_size=feature_size
        )
    else:
        raise ValueError('Invalid value for \'method\': {}'.format(method))


def _evaluate_line_shortest_path(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    graph,
    coordinate_system,  # pylint: disable=unused-argument
    feature_size
):
    """
    Evaluate the positions of a nodal line using the 'shortest path' method.
    """
    work_graph = graph.copy()
    distance_func = lambda x: x + x**2
    for edge in work_graph.edges:
        edge_attrs = work_graph.edges[edge]
        dist = edge_attrs[_DISTANCE_KEY]
        edge_attrs[_WEIGHT_KEY] = feature_size * distance_func(
            dist / feature_size
        )

    candidate_positions = set(work_graph.nodes)

    start_positions = set()
    first_pos = candidate_positions.pop()
    start_positions.add(first_pos)
    candidate_positions -= set(work_graph.neighbors(first_pos))

    result_graph = nx.Graph()
    result_graph.add_node(first_pos)

    high_degree_nodes = set()
    degree_one_nodes = set()

    while candidate_positions:

        end_pos = candidate_positions.pop()
        end_neighbours = set(work_graph.neighbors(end_pos)) | {end_pos}
        candidate_positions -= end_neighbours

        for start_pos in start_positions:
            # prioritize "crossings"
            _update_high_degree_nodes(
                work_graph, result_graph, high_degree_nodes
            )
            # prioritize ends
            _update_degree_one_nodes(
                work_graph, result_graph, degree_one_nodes
            )
            tmp_graph = work_graph.copy()

            start_neighbours = set(work_graph.neighbors(start_pos)
                                   ) | {start_pos}
            start_end_neighbours = start_neighbours | end_neighbours

            unweighted_path_length = nx.algorithms.shortest_path_length(
                tmp_graph,
                source=start_pos,
                target=end_pos,
            )
            if unweighted_path_length <= 4:
                single_path_only = True
                IDENTIFY_LOGGER.debug(
                    'Positions %s and %s are too close, calculating only one path.',
                    start_pos, end_pos
                )
            else:
                single_path_only = False
            for num_paths in range(_MAX_NUM_PATHS):
                if single_path_only and num_paths == 1:
                    break
                try:
                    path = nx.algorithms.shortest_path(
                        tmp_graph,
                        source=start_pos,
                        target=end_pos,
                        weight=_WEIGHT_KEY,
                    )
                except nx.NetworkXNoPath:
                    break

                if num_paths > 3:
                    print(path)
                new_neighbours = set()
                for node in path:
                    new_neighbours.update(tmp_graph.neighbors(node))
                    new_neighbours.add(node)
                candidate_positions -= new_neighbours

                nodes_to_remove = new_neighbours - start_end_neighbours
                assert nodes_to_remove or single_path_only

                edges = list(zip(path, path[1:]))
                result_graph.add_edges_from(edges)
                for edge in edges:
                    work_graph.edges[edge][_WEIGHT_KEY] = 0
                    tmp_graph.edges[edge][_WEIGHT_KEY] = 0

                tmp_graph.remove_nodes_from(nodes_to_remove)

            if num_paths > 0:  # pylint: disable=undefined-loop-variable
                IDENTIFY_LOGGER.debug(
                    'Found {} path(s) from point {} to point {}.'.format(
                        num_paths,  # pylint: disable=undefined-loop-variable
                        start_pos,
                        end_pos
                    )
                )
            else:
                IDENTIFY_LOGGER.warning(
                    'No paths to point {} found.'.format(end_pos)
                )
        start_positions.add(end_pos)

    return NodalLine(
        graph=result_graph, degree_count=_create_degree_count(result_graph)
    )


def _update_high_degree_nodes(graph, result_graph, high_degree_nodes):
    """
    Update the high degree nodes, and adjust the weight of the edges for new
    high degree nodes.
    """
    for node, deg in result_graph.degree:
        if deg > 2 and node not in high_degree_nodes:
            high_degree_nodes.add(node)
            _update_neighbour_weights(node, graph=graph, multiplier=0.5)


def _update_degree_one_nodes(graph, result_graph, degree_one_nodes):
    """
    Update the nodes of degree one. Decrease the weight of edges for
    new nodes of degree one, and increase the weight for those whose
    degree is no longer one.
    """
    current_deg_one_nodes = set(
        node for node, deg in result_graph.degree if deg == 1
    )
    new_nodes = current_deg_one_nodes - degree_one_nodes
    outdated_nodes = degree_one_nodes - current_deg_one_nodes
    for node in new_nodes:
        _update_neighbour_weights(node, graph=graph, multiplier=0.1)
    for node in outdated_nodes:
        _update_neighbour_weights(node, graph=graph, multiplier=10)
    degree_one_nodes.clear()
    degree_one_nodes.update(current_deg_one_nodes)


def _update_neighbour_weights(node, graph, multiplier):
    for nbr in graph.neighbors(node):
        graph.edges[(node, nbr)][_WEIGHT_KEY] *= multiplier


def _create_degree_count(graph):
    degree_counter = Counter([val for pos, val in graph.degree])
    degree_counter.pop(2, None)
    return degree_counter


def _evaluate_line_dominating_set(graph, coordinate_system, feature_size):
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
