# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""pytest configuration file for NodeFinder tests."""
# pylint: disable=unused-argument,protected-access,redefined-outer-name

import os
import json
import operator

import pytest

from score_fixtures import *  # noqa: F403


def pytest_addoption(parser):
    parser.addoption(
        "--no-plot-compare",
        action="store_true",
        help="disable comparing the generated plots",
    )


@pytest.fixture
def test_name(request):
    """Returns module_name.function_name for a given test"""
    return request.module.__name__ + "/" + request._parent_request._pyfuncitem.name


@pytest.fixture
def compare_data(request, test_name, scope="session"):
    """Returns a function which either saves some data to a file or (if that file exists already) compares it to pre-existing data using a given comparison function."""

    def inner(compare_fct, data, tag=None):
        full_name = test_name + (tag or "")
        val = request.config.cache.get(full_name, None)
        if val is None:
            request.config.cache.set(full_name, json.loads(json.dumps(data)))
            raise ValueError("Reference data does not exist.")
        val = json.loads(json.dumps(val))
        assert compare_fct(
            val, json.loads(json.dumps(data))
        )  # get rid of json-specific quirks

    return inner


@pytest.fixture
def compare_equal(compare_data):
    return lambda data, tag=None: compare_data(operator.eq, data, tag)


@pytest.fixture
def sample():
    """
    Fixture to get the path to the sample of a given name.
    """

    def inner(name):
        return os.path.join(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples"), name
        )

    return inner
