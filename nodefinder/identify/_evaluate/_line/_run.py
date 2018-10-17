"""
Defines the dispatcher function for evaluating a line, delegating to the specific
method implementation.
"""

from ._shortest_path import _evaluate_line_shortest_path
from ._dominating_set import _evaluate_line_dominating_set


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
