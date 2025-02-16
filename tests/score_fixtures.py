# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
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

    def inner(result, cutoff=None, additional_tag=""):
        num_fev = sum(res.num_fev for res in result.minimization_results)
        score(
            num_fev, tag=additional_tag + "num_fev", less_is_better=True, cutoff=cutoff
        )

    return inner


@export
@pytest.fixture
def score_nodal_points(score, score_num_fev):  # pylint: disable=redefined-outer-name
    """
    Fixture for scoring the result of a calculation which contains only nodal points.
    """

    def inner(
        result,
        exact_points,
        cutoff_accuracy=None,
        cutoff_coverage=None,
        additional_tag="",
    ):
        score_num_fev(result, additional_tag=additional_tag)
        distances = _get_distances(result=result, exact_points=exact_points)
        accuracy = np.max(np.min(distances, axis=-1))
        coverage = np.max(np.min(distances, axis=0))
        score(
            accuracy,
            less_is_better=True,
            cutoff=cutoff_accuracy,
            tag=additional_tag + "accuracy",
        )
        score(
            coverage,
            less_is_better=True,
            cutoff=cutoff_coverage,
            tag=additional_tag + "coverage",
        )

    return inner


def _get_distances(result, exact_points):
    return np.array(
        [
            result.coordinate_system.distance(res.pos, np.array(exact_points))
            for res in result.nodes.values()
        ]
    )


@export
@pytest.fixture
def score_nodal_line(score, score_num_fev):  # pylint: disable=redefined-outer-name
    """
    Fixture for scoring the accuracy and coverage of a nodal line.
    """

    def inner(
        result,
        *,
        dist_func=None,
        line_parametrization=None,
        cutoff_accuracy=None,
        cutoff_coverage=None,
        num_line_points=10**4,
    ):
        score_num_fev(result)

        if dist_func is not None:
            for res in result.nodes:
                if dist_func(res.pos) > 0.1:
                    print(dist_func(res.pos), res.pos)
            accuracy = max(dist_func(res.pos) for res in result.nodes)
            score(accuracy, tag="accuracy", cutoff=cutoff_accuracy, less_is_better=True)

        if line_parametrization is not None:
            line_points = np.array(
                [line_parametrization(x) for x in np.linspace(0, 1, num_line_points)]
            )
            distances = _get_distances(result=result, exact_points=line_points)
            coverage = np.max(np.min(distances, axis=0))
        score(coverage, tag="coverage", cutoff=cutoff_coverage, less_is_better=True)

    return inner


@export
@pytest.fixture
def score_nodal_surface(score, score_num_fev):  # pylint: disable=redefined-outer-name
    """
    Fixture for scoring the accuracy and coverage of a nodal surface.
    """

    def inner(
        result,
        *,
        dist_func=None,
        surface_parametrization=None,
        cutoff_accuracy=None,
        cutoff_coverage=None,
        num_line_points=100,
    ):
        score_num_fev(result)

        if dist_func is not None:
            for res in result.nodes:
                if dist_func(res.pos) > 0.1:
                    print(dist_func(res.pos), res.pos)
            accuracy = max(dist_func(res.pos) for res in result.nodes)
            score(accuracy, tag="accuracy", cutoff=cutoff_accuracy, less_is_better=True)

        if surface_parametrization is not None:
            surface_points = np.array(
                [
                    surface_parametrization(s, t)
                    for s in np.linspace(0, 1, num_line_points)
                    for t in np.linspace(0, 1, num_line_points)
                ]
            )
            distances = _get_distances(result=result, exact_points=surface_points)
            coverage = np.max(np.min(distances, axis=0))
        score(coverage, tag="coverage", cutoff=cutoff_coverage, less_is_better=True)

    return inner
