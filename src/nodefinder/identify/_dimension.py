# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the function for evaluating the dimension of a cluster of positions.
"""

import random
import warnings
import itertools
from collections import Counter

import numpy as np
import scipy.linalg as la
from scipy.special import binom
from fsc.export import export

EXACT_VALUES = {1: 0.5, 2: 8 / (9 * np.pi), 3: 27 * np.pi / 512}


@export
def calculate_dimension(
    *,
    graph,
    feature_size,
    coordinate_system,
    max_dim=3,
    min_pos_evaluations=5,
    min_neighbour_evaluations=10,
):
    """
    Calculate the dimension of a cluster of positions.

    Arguments
    ---------
    positions : list(tuple(float))
        The positions whose dimension should be evaluated.
    feature_size : float
        Distance between two nodal points at which they are considered to be
        distinct. This is used as a characteristic length scale to determine
        the dimension.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
    max_dim : int
        Upper limit for the dimension.
    min_pos_evaluations : int
        Minimum number of positions which are evaluated to estimate the dimension.
    min_neighbour_evaluations : int
        Minimum number of neighbour tuples which are evaluated per position.
    """
    positions_to_evaluate = list(graph.nodes)

    if min_pos_evaluations is not None:
        random.shuffle(positions_to_evaluate)

    dim_counter = Counter()
    for i, pos in enumerate(positions_to_evaluate):
        if i >= min_pos_evaluations:
            finished, dim = _check_count(dim_counter)
            if finished:
                return dim
        dim_counter.update(
            [
                _get_dimension(
                    pos,
                    coordinate_system=coordinate_system,
                    graph=graph,
                    feature_size=feature_size,
                    max_dim=max_dim,
                    min_neighbour_evaluations=min_neighbour_evaluations,
                )
            ]
        )
    finished, dim = _check_count(dim_counter)
    if not finished:
        warnings.warn("Inconclusive dimension count: {}".format(dim_counter))
        return None
    return dim


def _check_count(counter):
    """
    Get the most common dimension, and check if at least two thirds of the
    values agree.
    """
    dim, count = counter.most_common(1)[0]
    if count >= (2 / 3) * sum(counter.values()):
        return True, dim
    else:
        return False, dim


def _get_dimension(
    pos, *, graph, coordinate_system, feature_size, max_dim, min_neighbour_evaluations
):
    """
    Get the dimension from a given position.
    """
    for dim in range(1, max_dim + 1):
        if not _has_dimension(
            pos=pos,
            graph=graph,
            coordinate_system=coordinate_system,
            dim=dim,
            min_neighbour_evaluations=min_neighbour_evaluations,
            feature_size=feature_size,
        ):
            return dim - 1
    return max_dim


def _has_dimension(
    pos, *, graph, coordinate_system, dim, min_neighbour_evaluations, feature_size
):
    """
    Check if a position has at least the given dimension.
    """
    if len(pos) < dim:
        return 0
    neighbours = list(graph.neighbors(pos))
    if min_neighbour_evaluations is None:
        min_neighbour_evaluations = len(neighbours)

    try:
        limit_value = feature_size**dim * EXACT_VALUES[dim] / 2
    except KeyError:
        raise NotImplementedError(
            "Getting the {}-dimensional volume is not implemented".format(dim)
        )

    def draw_neighbour_tuple():
        """
        Generator to draw a tuple of neighbors, without repeating.
        """
        used_tuples = set()
        num_neighbours = len(neighbours)
        num_combinations = binom(num_neighbours, dim)
        # random sampling with check for uniqueness -- efficient for very
        # large number of possible combinations
        if num_combinations > 10 * min_neighbour_evaluations:
            while len(used_tuples) < (num_combinations / 2):
                val = tuple(sorted(random.sample(neighbours, k=dim)))

                if val not in used_tuples:
                    yield val
                    used_tuples.add(val)

        # switch to directly sampling from the remaining combinations
        # efficient when there are not so many (remaining) combinations
        neighbour_tuples = (
            set(tuple(sorted(n)) for n in itertools.combinations(neighbours, r=dim))
            - used_tuples
        )
        num_draws = 10
        while neighbour_tuples:
            vals = random.sample(
                list(neighbour_tuples), min(num_draws, len(neighbour_tuples))
            )
            yield from vals
            neighbour_tuples -= set(vals)

    results = []
    for i, neighbour_tuple in enumerate(draw_neighbour_tuple()):
        if i >= min_neighbour_evaluations and i > 1:
            avg = np.average(results)
            error_mean = np.sqrt(np.var(results) / (i - 1))
            if abs(limit_value - avg) / 2 > error_mean:
                return avg > limit_value
        results.append(
            _get_volume(
                pos=pos,
                neighbour_tuple=neighbour_tuple,
                coordinate_system=coordinate_system,
            )
        )
    return np.average(results) > limit_value


def _get_volume(pos, neighbour_tuple, coordinate_system):
    """
    Get the volume spanned by a position and the given neighbours.
    """
    connecting_vectors = [
        coordinate_system.connecting_vector(np.array(neighbour), np.array(pos))
        for neighbour in neighbour_tuple
    ]
    mat = np.array(connecting_vectors)
    svd = la.svd(mat, compute_uv=False)
    if len(svd) < len(neighbour_tuple):
        return 0
    return abs(np.prod(svd))
