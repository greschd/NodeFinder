import scipy.linalg as la
from fsc.export import export


@export
def periodic_distance(k1, k2):
    return la.norm([_periodic_distance_1d(a, b) for a, b in zip(k1, k2)])


def _periodic_distance_1d(k1, k2):
    k1 %= 1
    k2 %= 1
    return min((k1 - k2) % 1, (k2 - k1) % 1)
