import numpy as np
import scipy.linalg as la
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5

@subscribe_hdf5('nodefinder.coordinate_system')
class CoordinateSystem(HDF5Enabled):
    def __init__(self, *, limits, periodic=False):
        self._limits = np.array([sorted(x) for x in limits])
        self.periodic = periodic

    @property
    def _lower_limits(self):
        return self._limits[:, 0]

    @property
    def _upper_limits(self):
        return self._limits[:, 1]

    @property
    def size(self):
        return self._upper_limits - self._lower_limits

    def get_frac(self, pos, clip_limits=False):
        frac = (pos - self._lower_limits) / self._size
        if self.periodic:
            return frac % 1
        else:
            if clip_limits:
                return np.maximum(0, np.minimum(frac, 1))
            if not (np.all(0 <= frac) and np.all(frac <= 1)):
                raise ValueError(
                    "Position '{}' is out of bounds for limits '{}'".format(
                        pos, self._limits
                    )
                )
            return frac

    def distance(self, pos1, pos2):
        if self.periodic:
            return la.norm([
                self._periodic_distance_1d(p1, p2, s)
                for p1, p2, s in zip(pos1, pos2, self.size)
            ])
        return la.norm(pos2 - pos1)

    @staticmethod
    def _periodic_distance_1d(p1, p2, size):
        p1 %= size
        p2 %= size
        return min((p1 - p2) % size, (p2 - p2) % size)
