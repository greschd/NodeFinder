# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the dispatcher function for evaluating a line, delegating to the specific
method implementation.
"""

from collections import Counter

from ...result import NodalLine

from ._shortest_path import _evaluate_line_shortest_path
from ._dominating_set import _evaluate_line_dominating_set
from ._ballistic import _evaluate_line_ballistic

_METHOD_FUNC_LOOKUP = dict(
    shortest_path=_evaluate_line_shortest_path,
    dominating_set=_evaluate_line_dominating_set,
    ballistic=_evaluate_line_ballistic,
)


def _evaluate_line(graph, coordinate_system, feature_size, method="ballistic"):
    """
    Evaluate the positions of a nodal line.
    """

    try:
        method_func = _METHOD_FUNC_LOOKUP[method]
    except KeyError:
        raise ValueError("Invalid value for 'method': {}".format(method))
    result_graph = method_func(
        graph=graph, coordinate_system=coordinate_system, feature_size=feature_size
    )
    return NodalLine(
        graph=result_graph, degree_count=_create_degree_count(result_graph)
    )


def _create_degree_count(graph):
    degree_counter = Counter([val for pos, val in graph.degree])
    degree_counter.pop(2, None)
    return degree_counter
