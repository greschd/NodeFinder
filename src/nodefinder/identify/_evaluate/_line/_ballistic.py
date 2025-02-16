# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Implements the 'ballistic' method for evaluating the line, which starts at a
specific point and then tries going into the most populated direction. The
angle to the previous step and distance to the next node are also taken into
consideration in choosing each step.
"""

import numpy as np
import scipy.linalg as la
import networkx as nx

from ..._logging import IDENTIFY_LOGGER
from ..._cluster import _DISTANCE_KEY

# Multipliers for the weight which determines the next point to pick
# These are the 'magic numbers' which determine how well the
# line evaluation performs.
_MULTIPLIER_HIGH_DEGREE = 1
_MULTIPLIER_LOW_DEGREE = 1
_MULTIPLIER_EXISTING_NODE = 0.2
_MULTIPLIER_DISTANCE = 5


def _evaluate_line_ballistic(*, graph, coordinate_system, feature_size):
    """
    Runs the line evaluation using the 'ballistic' method.
    """
    runner = _BallisticLineImpl(
        graph=graph, coordinate_system=coordinate_system, feature_size=feature_size
    )
    runner.run()
    return runner.result_graph


class _BallisticLineImpl:
    """
    Implementation class for the 'ballistic' line evaluation.
    """

    def __init__(self, *, graph, coordinate_system, feature_size):
        self.graph = graph
        self.coordinate_system = coordinate_system
        self.feature_size = feature_size
        self.result_graph = nx.Graph()
        self.evaluated_end_nodes = set()

    def run(self):
        """
        Runs the ballistic line evaluation.
        """
        IDENTIFY_LOGGER.debug("Starting ballistic line evaluation.")
        # node = next(iter(self.graph.nodes))
        # self.find_loop(starting_node=node, initial_direction=None)
        while not nx.is_dominating_set(self.graph, self.result_graph.nodes):
            # First, try starting from an end node
            end_nodes = {node for node, deg in self.result_graph.degree if deg == 1}
            new_end_nodes = end_nodes - self.evaluated_end_nodes
            if new_end_nodes:
                node = new_end_nodes.pop()
                nbr = next(self.result_graph.neighbors(node))
                delta = self.coordinate_system.connecting_vector(
                    np.array(node), np.array(nbr)
                )
                # go in the opposite direction from the previous path
                initial_direction = -(delta / la.norm(delta))
                self.find_loop(starting_node=node, initial_direction=initial_direction)
                self.evaluated_end_nodes.add(node)
            # Otherwise, pick an arbitrary starting node not in the vicinity of the result graph
            else:
                neighbor_graph = self.graph.edge_subgraph(
                    self.graph.edges(self.result_graph.nodes)
                )
                candidate_nodes = set(self.graph.nodes) - set(neighbor_graph.nodes)
                node = candidate_nodes.pop()
                self.find_loop(starting_node=node, initial_direction=None)

    def find_loop(self, *, starting_node, initial_direction):
        """
        Find a loop from the given starting node, going (roughly) in the given
        direction. Stops when either the loop is closed (two nodes within the
        result graph found), or there are no more available steps.
        """
        IDENTIFY_LOGGER.debug(
            "Searching loop starting from node %s, with direction %s",
            starting_node,
            initial_direction,
        )
        node = starting_node
        previous_direction = initial_direction
        self.result_graph.add_node(node)
        on_graph = False
        while True:
            new_node, new_direction = self.get_next_pos(
                node=node,
                previous_direction=previous_direction,
            )
            if new_node is None:
                self.evaluated_end_nodes.add(node)
                IDENTIFY_LOGGER.debug(
                    "Loop search finished -- no more nodes in given direction. Current node: %s, direction: %s",
                    node,
                    previous_direction,
                )
                assert previous_direction is not None
                break
            if new_node in self.result_graph:
                if on_graph:
                    IDENTIFY_LOGGER.debug(
                        "Loop search finished -- reached existing result nodes."
                    )
                    break
                on_graph = True
            else:
                on_graph = False
            self.result_graph.add_edge(node, new_node)
            node = new_node
            previous_direction = new_direction

    def get_next_pos(self, *, node, previous_direction):  # pylint: disable=too-many-locals
        """
        Determine the next node based on the current node and previous step
        direction.
        """
        neighbors_all = list(self.graph.neighbors(node))

        deltas_all = self.coordinate_system.connecting_vector(
            np.array(node), np.array(neighbors_all)
        )
        delta_norms_all = la.norm(deltas_all, axis=-1)

        neighbors = []
        unit_vecs = []
        for nbr, delta, delta_norm in zip(neighbors_all, deltas_all, delta_norms_all):
            if delta_norm > 0:
                neighbors.append(nbr)
                unit_vecs.append(delta / delta_norm)
        unit_vecs = np.array(unit_vecs)

        if previous_direction is not None:
            direction_angles = unit_vecs @ previous_direction
        else:
            direction_angles = np.ones(len(unit_vecs))
        positive_angle_idx = np.where(direction_angles > 0)[0]
        if positive_angle_idx.size == 0:
            return None, None
        candidate_nodes = [neighbors[i] for i in positive_angle_idx]

        candidate_unit_vecs = unit_vecs[positive_angle_idx]
        angle_weight_matrix = candidate_unit_vecs @ candidate_unit_vecs.T

        weights = np.sum(np.maximum(angle_weight_matrix, 0), axis=-1)
        weights *= direction_angles[positive_angle_idx]

        # prefer nodes which are already in the result graph
        in_result_graph = np.array(
            [nbr in self.result_graph.nodes for nbr in candidate_nodes]
        )
        weights *= 1 + _MULTIPLIER_EXISTING_NODE * in_result_graph.astype(int)

        node_degrees = dict(self.result_graph.degree)
        # prefer nodes which have a higher degree
        has_higher_degree = np.array(
            [node_degrees.get(nbr, 0) > 2 for nbr in candidate_nodes]
        )
        weights *= 1 + _MULTIPLIER_HIGH_DEGREE * has_higher_degree.astype(int)
        # prefer nodes which have degree one
        has_low_degree = np.array(
            [node_degrees.get(nbr, 0) == 1 for nbr in candidate_nodes]
        )
        weights *= 1 + _MULTIPLIER_LOW_DEGREE * has_low_degree.astype(int)

        # prefer nodes which are close to the current one
        distances_normalized = (
            np.array(
                [
                    self.graph.edges[(node, neighbors[i])][_DISTANCE_KEY]
                    for i in positive_angle_idx
                ]
            )
            / self.feature_size
        )
        weights /= 1 + _MULTIPLIER_DISTANCE * distances_normalized

        # avoid nodes which are much too close
        too_close = distances_normalized < 1e-3
        weights *= 1 - too_close.astype(int)

        idx = positive_angle_idx[np.argmax(weights)]
        chosen_candidate = neighbors[idx]

        return chosen_candidate, unit_vecs[idx]
