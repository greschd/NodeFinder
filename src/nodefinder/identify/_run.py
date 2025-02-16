# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the function which runs the identify step.
"""

from fsc.export import export

from ..search.result import ControllerState
from ..search._controller import _DIST_CUTOFF_FACTOR

from .result import IdentificationResult, IdentificationResultContainer
from ._cluster import create_clusters
from ._dimension import calculate_dimension
from ._evaluate import evaluate_cluster
from ._logging import IDENTIFY_LOGGER


@export
def run(result, feature_size=None, evaluate_line_method="ballistic"):
    """Identify the nodal clusters from a :func:`.search.run` result.

    Arguments
    ---------
    result : SearchResultContainer
        Result of the search step.
    feature_size : float, optional
        Distance between two nodal points at which they are considered distinct.
        Uses the ``feature_size`` used in the search step by default.
    """
    if feature_size is None:
        feature_size = result.dist_cutoff * _DIST_CUTOFF_FACTOR
    if isinstance(result, ControllerState):
        result = result.result
    positions = [node.pos for node in result.nodes]
    coordinate_system = result.coordinate_system
    return run_from_positions(
        positions=positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size,
        evaluate_line_method=evaluate_line_method,
    )


@export
def run_from_positions(
    positions, *, coordinate_system, feature_size, evaluate_line_method="shortest_path"
):
    """Identify the nodal clusters from a list of positions.

    Arguments
    ---------
    positions : list(tuple(float))
        The list of nodal positions.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
    feature_size : float
        Distance between two nodal points at which they are considered distinct.
    """
    IDENTIFY_LOGGER.debug("Calculating clusters.")
    clusters = create_clusters(
        positions, coordinate_system=coordinate_system, feature_size=feature_size
    )
    results = []
    for graph in clusters:
        IDENTIFY_LOGGER.debug("Calculating cluster dimension.")
        dim = calculate_dimension(
            graph=graph, coordinate_system=coordinate_system, feature_size=feature_size
        )
        IDENTIFY_LOGGER.debug("Evaluating result with dimension {}.".format(dim))
        res = IdentificationResult(
            positions=list(graph.nodes),
            dimension=dim,
            shape=evaluate_cluster(
                graph=graph,
                dim=dim,
                coordinate_system=coordinate_system,
                feature_size=feature_size,
                evaluate_line_method=evaluate_line_method,
            ),
        )
        results.append(res)
    return IdentificationResultContainer(
        coordinate_system=coordinate_system, feature_size=feature_size, results=results
    )
