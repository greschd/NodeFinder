"""pytest configuration file for NodeFinder tests."""
# pylint: disable=unused-argument,protected-access,redefined-outer-name

import json
import operator

import pytest
import numpy as np


@pytest.fixture
def test_name(request):
    """Returns module_name.function_name for a given test"""
    return request.module.__name__ + '/' + request._parent_request._pyfuncitem.name


@pytest.fixture
def compare_data(request, test_name, scope="session"):
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""

    def inner(compare_fct, data, tag=None):
        full_name = test_name + (tag or '')
        val = request.config.cache.get(full_name, None)
        if val is None:
            request.config.cache.set(full_name, json.loads(json.dumps(data)))
            raise ValueError('Reference data does not exist.')
        else:
            val = json.loads(json.dumps(val))
            assert compare_fct(val, json.loads(json.dumps(data))
                               )  # get rid of json-specific quirks

    return inner


@pytest.fixture
def compare_equal(compare_data):
    return lambda data, tag=None: compare_data(operator.eq, data, tag)


@pytest.fixture
def score_num_fev(score):
    def inner(result, cutoff=None):
        num_fev = sum(res.num_fev for res in result.minimization_results)
        score(num_fev, tag='num_fev', less_is_better=True, cutoff=cutoff)

    return inner


@pytest.fixture
def score_nodal_points(score):
    def inner(
        result, exact_points, cutoff_accuracy=None, cutoff_coverage=None
    ):
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
            tag='nodal_point_accuracy'
        )
        score(
            coverage,
            less_is_better=True,
            cutoff=cutoff_coverage,
            tag='nodal_point_coverage'
        )

    return inner
