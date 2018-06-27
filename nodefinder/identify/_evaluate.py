from types import SimpleNamespace

import numpy as np


class NodalPoint(SimpleNamespace):
    def __init__(self, position):
        self.position = position


def evaluate_cluster(positions, dim, coordinate_system, neighbour_mapping):
    if dim == 0:
        return _evaluate_point(
            positions=positions, coordinate_system=coordinate_system
        )
    # elif dim == 1:
    # return None
    else:
        return None


def _evaluate_point(positions, coordinate_system):
    return NodalPoint(
        position=coordinate_system.
        average([np.array(pos) for pos in positions])
    )

    # print(len(clusters))
    # print(clusters, neighbour_mapping)
