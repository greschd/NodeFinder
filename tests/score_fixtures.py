"""
Defines the fixtures for scoring results of NodeFinder calculations.
"""

import pytest
import numpy as np
from fsc.export import export


@export
@pytest.fixture
def score_num_fev(score):
    """
    Fixture for scoring the number of function evaluations that were performed.
    """

    def inner(result, cutoff=None, additional_tag=''):
        num_fev = sum(res.num_fev for res in result.minimization_results)
        score(
            num_fev,
            tag=additional_tag + 'num_fev',
            less_is_better=True,
            cutoff=cutoff
        )

    return inner


@export
@pytest.fixture
def score_nodal_points(score, score_num_fev):  # pylint: disable=redefined-outer-name
    """
    Fixture for scoring the result of a calculation which contains only nodal points.
    """

    def inner(  # pylint: disable=missing-docstring
        result,
        exact_points,
        cutoff_accuracy=None,
        cutoff_coverage=None,
        additional_tag=''
    ):
        assert len(result.nodes) >= len(exact_points)
        score_num_fev(result, additional_tag=additional_tag)
        distances = np.array([[
            result.coordinate_system.distance(res.pos, exact_pos)
            for exact_pos in exact_points
        ] for res in result.nodes.values()])
        accuracy = np.max(np.min(distances, axis=-1))
        coverage = np.max(np.min(distances, axis=0))
        score(
            accuracy,
            less_is_better=True,
            cutoff=cutoff_accuracy,
            tag=additional_tag + 'nodal_point_accuracy'
        )
        score(
            coverage,
            less_is_better=True,
            cutoff=cutoff_coverage,
            tag=additional_tag + 'nodal_point_coverage'
        )

    return inner
