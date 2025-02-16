# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the coordinate system class.
"""

import numpy as np
import scipy.linalg as la
from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5


@export
@subscribe_hdf5("nodefinder.coordinate_system")
class CoordinateSystem(SimpleHDF5Mapping):
    """
    Defines a "box" coordinate system, which is used to calculate the distances
    between points, and map between fractional and real coordinates.

    Attributes
    ----------
    limits : numpy.ndarray
        Limits of the coordinates, given as a the minimum and
        maximum value for each dimension.
    periodic : bool
        Determies if periodic boundary conditions are used.
    dim : int
        Dimension of the coordinate system.
    size : numpy.ndarray
        Size of the coordinate system in each dimension.
    """

    HDF5_ATTRIBUTES = ["limits", "periodic"]

    def __init__(self, *, limits, periodic=True):
        self.limits = np.array([sorted(x) for x in limits])
        self.periodic = periodic
        self.dim = len(self.limits)
        self.size = self._upper_limits - self._lower_limits

    def __repr__(self):
        return "CoordinateSystem(limits={0.limits!r}, periodic={0.periodic!r})".format(
            self
        )

    @property
    def _lower_limits(self):
        return self.limits[:, 0]

    @property
    def _upper_limits(self):
        return self.limits[:, 1]

    def get_frac(self, pos):
        """
        Get the fractional coordinates from the real position.
        """
        frac = (pos - self._lower_limits) / self.size
        if self.periodic:
            frac %= 1
        return frac

    def get_pos(self, frac):
        """
        Get the real position from the fractional coordinates.
        """
        return (frac * self.size) + self._lower_limits

    def distance(self, pos1, pos2):
        """
        Get the distance between two positions.
        """
        if self.periodic:
            delta = (pos2 - pos1) % self.size
            delta = np.minimum(self.size - delta, delta)  # pylint: disable=assignment-from-no-return,useless-suppression
        else:
            delta = pos2 - pos1
        return la.norm(delta, axis=-1)

    def connecting_vector(self, pos1, pos2):
        """
        Get the shortest vector connecting two positions.
        """
        if not self.periodic:
            return pos2 - pos1
        else:
            delta = (pos2 - pos1) % self.size
            delta_negative = delta - self.size
            res = np.where(-delta_negative < delta, delta_negative, delta)
            assert np.all(self.distance((res + pos1), pos2) < 1e-8)
            return res

    def average(self, positions):
        """
        Get the average position from a list of positions.
        """
        if not self.periodic:
            return np.average(positions, axis=0)
        else:
            origin = positions[0]
            deltas = [self.connecting_vector(origin, pos) for pos in positions]
            return (origin + np.average(deltas, axis=0)) % self.size

    def normalize_position(self, pos):
        """
        Normalize the position by mapping it into the limits for periodic
        boundary conditions for the periodic case.
        """
        if self.periodic:
            return ((pos - self._lower_limits) % self.size) + self._lower_limits
        else:
            return pos
