# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the function which evaluates the shape of a cluster by delegating to the
implementation for the given cluster dimension.
"""

import warnings

from fsc.export import export

from ._point import _evaluate_point
from ._line import _evaluate_line

_WEIGHT_KEY = "_weight"


@export
def evaluate_cluster(
    graph, dim, coordinate_system, feature_size, evaluate_line_method="shortest_path"
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
            warnings.warn("Could not identify line: {}".format(exc))
    else:
        return None
