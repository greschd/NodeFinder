# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines a helper function for generating a starting mesh.
"""

import itertools

import numpy as np


def _generate_mesh_simplices(*, limits, mesh_size, periodic=False, skip_origin=False):
    """
    Generate the starting simplices for given limits and mesh size.
    """
    dim = len(limits)
    if len(mesh_size) != dim:
        raise ValueError("Inconsistent dimensions: {}, {}".format(dim, len(mesh_size)))
    vertices = list(
        itertools.product(
            *[
                np.linspace(lower, upper, m, endpoint=not periodic)
                for (lower, upper), m in zip(limits, mesh_size)
            ]
        )
    )
    if skip_origin:
        vertices = [v for v in vertices if not np.allclose(v, 0)]
    size = np.array([upper - lower for lower, upper in limits])
    simplex_distances = size / (2 * np.array(mesh_size))
    simplex_stencil = np.zeros(shape=(dim + 1, dim))
    for i, dist in enumerate(simplex_distances):
        simplex_stencil[i + 1][i] = dist
    return [v + simplex_stencil for v in vertices]
