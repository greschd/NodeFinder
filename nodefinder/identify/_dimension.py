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
    num_neighbour_evaluations=25,
):
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
        warnings.warn(
            'Inconclusive dimension count: {}'.format(dim_counter)
        )
        return None
    return dim


def _get_dimension(
    pos, *, neighbours, coordinate_system, feature_size, max_dim, num_neighbour_evaluations
):
    for dim in range(1, max_dim + 1):
        volume = _get_volume(
            pos,
            neighbours,
            coordinate_system=coordinate_system,
            dim=dim,
            num_neighbour_evaluations=num_neighbour_evaluations
        )
        volume_normalized = volume / (0.5 * feature_size)**dim
        if volume_normalized < 1:
            return dim - 1
    return max_dim


def _get_volume(pos, neighbours, *, coordinate_system, dim, num_neighbour_evaluations):
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
            coordinate_system.connecting_vector(np.array(neighbour), np.array(pos))
            for neighbour in neighbour_tuple
        ]
        mat = np.array(connecting_vectors)
        results.append(abs(np.product(la.svd(mat, compute_uv=False))))
    return np.average(results)
