# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Implements line evaluation with the shortest path method.
"""

import networkx as nx

from ..._cluster import _DISTANCE_KEY
from ..._logging import IDENTIFY_LOGGER

_WEIGHT_KEY = "_weight"
_MAX_NUM_PATHS = 20


def _evaluate_line_shortest_path(  # pylint: disable=too-many-locals,too-many-statements,too-many-branches
    *,
    graph,
    coordinate_system,  # pylint: disable=unused-argument
    feature_size,
):
    """
    Evaluate the positions of a nodal line using the 'shortest path' method.
    """
    work_graph = graph.copy()
    distance_func = lambda x: x + x**2
    for edge in work_graph.edges:
        edge_attrs = work_graph.edges[edge]
        dist = edge_attrs[_DISTANCE_KEY]
        edge_attrs[_WEIGHT_KEY] = feature_size * distance_func(dist / feature_size)

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
            _update_high_degree_nodes(work_graph, result_graph, high_degree_nodes)
            # prioritize ends
            _update_degree_one_nodes(work_graph, result_graph, degree_one_nodes)
            tmp_graph = work_graph.copy()

            start_neighbours = set(work_graph.neighbors(start_pos)) | {start_pos}
            start_end_neighbours = start_neighbours | end_neighbours

            unweighted_path_length = nx.algorithms.shortest_path_length(
                tmp_graph,
                source=start_pos,
                target=end_pos,
            )
            if unweighted_path_length <= 4:
                single_path_only = True
                IDENTIFY_LOGGER.debug(
                    "Positions %s and %s are too close, calculating only one path.",
                    start_pos,
                    end_pos,
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

            if num_paths > 0:
                IDENTIFY_LOGGER.debug(
                    "Found {} path(s) from point {} to point {}.".format(
                        num_paths, start_pos, end_pos
                    )
                )
            else:
                IDENTIFY_LOGGER.warning("No paths to point {} found.".format(end_pos))
        start_positions.add(end_pos)

    return result_graph


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
    current_deg_one_nodes = set(node for node, deg in result_graph.degree if deg == 1)
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
