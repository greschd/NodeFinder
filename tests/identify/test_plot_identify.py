# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the identify plot functions.
"""

import os

import pytest
import nodefinder as nf

from plottest_helpers import *  # pylint: disable=import-error


@pytest.mark.parametrize(
    'sample_name', ['line', 'point', 'surface', 'two_lines', 'line_periodic']
)
def test_result(sample, sample_name, assert_image_equal):
    """
    Test the 'identify.plot.result' plotting function
    """
    res = nf.io.load(sample(os.path.join('identify', sample_name + '.hdf5')))
    nf.identify.plot.result(res)
    assert_image_equal('identify:' + sample_name)
