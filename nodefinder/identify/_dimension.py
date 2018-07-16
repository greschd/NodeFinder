"""
Defines the function for evaluating the dimension of a cluster of positions.
"""

import random
import warnings
import itertools
from collections import Counter

import numpy as np
import scipy.linalg as la
from fsc.export import export


@export
def calculate_dimension(
    *,
    positions,
    neighbour_mapping,
    feature_size,
    coordinate_system,
    max_dim=3,
    num_pos_evaluations=10,
    num_neighbour_evaluations=25
):
    """
    Calculate the dimension of a cluster of positions.

    Arguments
    ---------
    positions : list(tuple(float))
        The positions whose dimension should be evaluated.
    neighbour_mapping : collections.abc.Mapping
        Contains a list of neighbours for each position.
    feature_size : float
        Distance between two nodal points at which they are considered to be
        distinct. This is used as a characteristic length scale to determine
        the dimension.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
    max_dim : int
        Upper limit for the dimension.
    num_pos_evaluations : int
        Number of positions which are evaluated to estimate the dimension.
    num_neighbour_evaluations : int
        Number of neighbour tuples which are evaluated per position.
    """
    if num_pos_evaluations is None:
        positions_to_evaluate = positions
    else:
        positions_to_evaluate = random.sample(
            positions, min(len(positions), num_pos_evaluations)
        )
    dim_counter = Counter()
    for pos in positions_to_evaluate:
        dim_counter.update([
            _get_dimension(
                pos,
                coordinate_system=coordinate_system,
                neighbours=neighbour_mapping[pos],
                feature_size=feature_size,
                max_dim=max_dim,
                num_neighbour_evaluations=num_neighbour_evaluations
            )
        ])
    dim, count = dim_counter.most_common(1)[0]
    if count < (2 / 3) * len(positions_to_evaluate):
        warnings.warn('Inconclusive dimension count: {}'.format(dim_counter))
        return None
    return dim


def _get_dimension(
    pos, *, neighbours, coordinate_system, feature_size, max_dim,
    num_neighbour_evaluations
):
    """
    Get the dimension from a given positions.
    """
    for dim in range(1, max_dim + 1):
        volume = _get_volume(
            pos,
            neighbours,
            coordinate_system=coordinate_system,
            dim=dim,
            num_neighbour_evaluations=num_neighbour_evaluations
        )
        volume_normalized = volume / (feature_size / 2)**dim
        if volume_normalized < 0.5:
            return dim - 1
    return max_dim


def _get_volume(
    pos, neighbours, *, coordinate_system, dim, num_neighbour_evaluations
):
    """
    Get an estimate for the average volume spanned by a position and its neighbours.
    """
    neighbour_pos = [n.pos for n in neighbours]
    if num_neighbour_evaluations is None:
        num_neighbour_evaluations = len(neighbours)
    else:
        random.shuffle(neighbour_pos)
    results = []
    for neighbour_tuple, _ in zip(
        itertools.combinations(neighbour_pos, r=dim),
        range(num_neighbour_evaluations)
    ):
        connecting_vectors = [
            coordinate_system.connecting_vector(
                np.array(neighbour), np.array(pos)
            ) for neighbour in neighbour_tuple
        ]
        mat = np.array(connecting_vectors)
        results.append(abs(np.product(la.svd(mat, compute_uv=False))))
    return np.average(results)
