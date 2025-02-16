# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the fake potential class.
"""


class FakePotential:
    """
    Defines the fake potential used to repel the minimization from the existing
    nodes.

    Arguments
    ---------
    result : SearchResultContainer
        The existing results from which the minimization should be repelled.
    width : float
        Distance from existing nodes at which the fake potential should start.
    """

    def __init__(self, result, width):
        self.result = result
        self.width = width

    def __call__(self, pos):  # pylint: disable=missing-function-docstring
        if any(
            dist < self.width for dist in self.result.get_all_neighbour_distances(pos)
        ):
            return float("inf")
        return 0
