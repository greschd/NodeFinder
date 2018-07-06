import os

import pytest
import nodefinder as nf

from plottest_helpers import *  # pylint: disable=unused-wildcard-import


@pytest.mark.parametrize(
    'sample_name', ['line', 'point', 'surface', 'two_lines']
)
def test_points(sample, sample_name, assert_image_equal):  # pylint: disable=redefined-outer-name
    res = nf.io.load(sample(os.path.join('search', sample_name + '.hdf5')))
    nf.search.plot.points(res)
    assert_image_equal('search:' + sample_name)
