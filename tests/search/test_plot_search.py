# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines tests for the search plotting functions.
"""

import os

import pytest
import nodefinder as nf

from plottest_helpers import *  # noqa: F403


@pytest.mark.parametrize("sample_name", ["line", "point", "surface", "two_lines"])
def test_points(sample, sample_name, assert_image_equal):
    """
    Test for the 'search.plot.points' function.
    """
    res = nf.io.load(sample(os.path.join("search", sample_name + ".hdf5")))
    nf.search.plot.points(res)
    assert_image_equal("search:" + sample_name)
