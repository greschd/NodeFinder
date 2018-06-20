import numpy as np
import scipy.linalg as la
from fsc.export import export
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5


@export
@subscribe_hdf5('nodefinder.coordinate_system')
class CoordinateSystem(HDF5Enabled):
    def __init__(self, *, limits, periodic=False):
        self.limits = np.array([sorted(x) for x in limits])
        self.periodic = periodic

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

    @property
    def size(self):
        return self._upper_limits - self._lower_limits

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
            return la.norm([
                self._periodic_distance_1d(p1, p2, s)
                for p1, p2, s in zip(pos1, pos2, self.size)
            ])
        return la.norm(pos2 - pos1)

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

    @staticmethod
    def _periodic_distance_1d(x, y, size):
        x %= size
        y %= size
        return min((x - y) % size, (y - x) % size)

    def to_hdf5(self, hdf5_handle):
        hdf5_handle['limits'] = self.limits
        hdf5_handle['periodic'] = self.periodic

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            limits=hdf5_handle['limits'].value,
            periodic=hdf5_handle['periodic'].value
        )
