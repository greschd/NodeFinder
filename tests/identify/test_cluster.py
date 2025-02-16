# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests for the clustering algorithm.
"""

import pytest

import nodefinder as nf
from nodefinder.search._controller import _DIST_CUTOFF_FACTOR
from nodefinder.identify._cluster import create_clusters


@pytest.mark.parametrize(
    "sample_name, num_clusters",
    [
        ("search/two_lines.hdf5", 2),
        ("search/point.hdf5", 1),
        ("search/line.hdf5", 1),
        ("search/surface.hdf5", 1),
    ],
)
def test_clustering(sample, sample_name, num_clusters):
    """
    Test that the positions given in the sample search outputs are correctly
    clustered.
    """
    search_result = nf.io.load(sample(sample_name))

    positions = [res.pos for res in search_result.nodes]
    coordinate_system = search_result.coordinate_system
    feature_size = search_result.dist_cutoff * _DIST_CUTOFF_FACTOR

    clusters = create_clusters(
        positions=positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size,
    )
    assert len(clusters) == num_clusters
