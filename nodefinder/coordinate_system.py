import numpy as np
import scipy.linalg as la
from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5


@export
@subscribe_hdf5('nodefinder.coordinate_system')
class CoordinateSystem(SimpleHDF5Mapping):
    HDF5_ATTRIBUTES = ['limits', 'periodic']

    def __init__(self, *, limits, periodic=True):
        self.limits = np.array([sorted(x) for x in limits])
        self.periodic = periodic
        self.dim = len(self.limits)
        self.size = self._upper_limits - self._lower_limits

    def __repr__(self):
        return 'CoordinateSystem(limits={0.limits!r}, periodic={0.periodic!r})'.format(
            self
        )

    @property
    def _lower_limits(self):
        return self.limits[:, 0]

    @property
    def _upper_limits(self):
        return self.limits[:, 1]

    def get_frac(self, pos, clip_limits=False):
        frac = (pos - self._lower_limits) / self.size
        if self.periodic:
            return frac % 1
        else:
            if clip_limits:
                return np.maximum(0, np.minimum(frac, 1))
            if not (np.all(frac >= 0) and np.all(frac <= 1)):
                raise ValueError(
                    "Position '{}' is out of bounds for limits '{}'".format(
                        pos, self.limits
                    )
                )
            return frac

    def get_pos(self, frac):
        return (frac * self.size) + self._lower_limits

    def distance(self, pos1, pos2):
        if self.periodic:
            delta = (pos2 - pos1) % self.size
            delta = np.minimum(self.size - delta, delta)
        else:
            delta = pos2 - pos1
        return la.norm(delta, axis=-1)

    def connecting_vector(self, pos1, pos2):
        if not self.periodic:
            return pos2 - pos1
        else:
            delta = (pos2 - pos1) % self.size
            delta_negative = delta - self.size
            res = np.where(-delta_negative < delta, delta_negative, delta)
            assert np.allclose((res + pos1) % self.size, pos2 % self.size)
            return res

    def average(self, positions):
        if not self.periodic:
            return np.average(positions, axis=0.)
        else:
            origin = positions[0]
            deltas = [self.connecting_vector(origin, pos) for pos in positions]
            return (origin + np.average(deltas, axis=0)) % self.size

    def normalize_position(self, pos):
        if self.periodic:
            return ((pos - self._lower_limits) %
                    self.size) + self._lower_limits
        else:
            if np.all(pos >= self._lower_limits
                      ) and np.all(pos <= self._upper_limits):
                return pos
            raise ValueError(
                "Position '{}' is not within the limits '{}'".format(
                    pos, self.limits
                )
            )
