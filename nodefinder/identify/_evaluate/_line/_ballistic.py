"""
Implements the 'ballistic' method for evaluating the line, which starts at a
specific point and then tries going into the most populated direction. The
angle to the previous step and distance to the next node are also taken into
consideration in choosing each step.
"""

import numpy as np
import scipy.linalg as la
import networkx as nx

from ..._cluster import _DISTANCE_KEY

# Multipliers for the weight which determines the next point to pick
# These are the 'magic numbers' which determine how well the
# line evaluation performs.
_MULTIPLIER_DEGREE = 1
_MULTIPLIER_EXISTING_NODE = 1
_MULTIPLIER_DISTANCE = 5


def _evaluate_line_ballistic(*, graph, coordinate_system, feature_size):
    """
    Runs the line evaluation using the 'ballistic' method.
    """
    runner = _BallisticLineImpl(
        graph=graph,
        coordinate_system=coordinate_system,
        feature_size=feature_size
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
        # node = next(iter(self.graph.nodes))
        # self.find_loop(starting_node=node, initial_direction=None)
        while not nx.is_dominating_set(self.graph, self.result_graph.nodes):
            # First, try starting from an end node
            end_nodes = {
                node
                for node, deg in self.result_graph.degree if deg == 1
            }
            new_end_nodes = end_nodes - self.evaluated_end_nodes
            if new_end_nodes:
                node = new_end_nodes.pop()
                nbr = next(self.result_graph.neighbors(node))
                delta = self.coordinate_system.connecting_vector(
                    np.array(node), np.array(nbr)
                )
                # go in the opposite direction from the previous path
                initial_direction = -(delta / la.norm(delta))
                self.find_loop(
                    starting_node=node, initial_direction=initial_direction
                )
                self.evaluated_end_nodes.add(node)
            # Otherwise, pick an arbitrary starting node not in the vicinity of the result graph
            else:
                neighbor_graph = self.graph.edge_subgraph(
                    self.graph.edges(self.result_graph.nodes)
                )
                candidate_nodes = set(self.graph.nodes
                                      ) - set(neighbor_graph.nodes)
                node = candidate_nodes.pop()
                self.find_loop(starting_node=node, initial_direction=None)

    def find_loop(self, *, starting_node, initial_direction):
        """
        Find a loop from the given starting node, going (roughly) in the given
        direction. Stops when either the loop is closed (two nodes within the
        result graph found), or there are no more available steps.
        """
        node = starting_node
        previous_direction = initial_direction
        self.result_graph.add_node(node)
        on_graph = False
        while True:
            new_node, previous_direction = self.get_next_pos(
                node=node,
                previous_direction=previous_direction,
            )
            if new_node is None:
                self.evaluated_end_nodes.add(node)
                break
            elif new_node in self.result_graph:
                if on_graph:
                    break
                on_graph = True
            else:
                on_graph = False
            self.result_graph.add_edge(node, new_node)
            node = new_node

    def get_next_pos(self, *, node, previous_direction):
        """
        Determine the next node based on the current node and previous step
        direction.
        """
        neighbors = list(self.graph.neighbors(node))

        deltas = self.coordinate_system.connecting_vector(
            np.array(node), np.array(neighbors)
        )
        unit_vecs = (deltas.T / la.norm(deltas, axis=-1)).T
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
        in_result_graph = np.array([
            nbr in self.result_graph.nodes for nbr in candidate_nodes
        ])
        weights *= 1 + _MULTIPLIER_EXISTING_NODE * in_result_graph.astype(int)

        node_degrees = dict(self.result_graph.degree)
        # prefer nodes which have a higher degree
        has_higher_degree = np.array([
            node_degrees.get(nbr, 0) > 2 for nbr in candidate_nodes
        ])
        weights *= 1 + _MULTIPLIER_DEGREE * has_higher_degree.astype(int)

        # prefer nodes which are close to the current one
        distances_normalized = np.array([
            self.graph.edges[(node, neighbors[i])][_DISTANCE_KEY]
            for i in positive_angle_idx
        ]) / self.feature_size
        weights /= (1 + _MULTIPLIER_DISTANCE * distances_normalized)

        idx = positive_angle_idx[np.argmax(weights)]
        chosen_candidate = neighbors[idx]

        return chosen_candidate, unit_vecs[idx]
