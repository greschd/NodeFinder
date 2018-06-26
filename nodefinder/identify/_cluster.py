import itertools
from collections import namedtuple

import numpy as np

Neighbour = namedtuple('Neighbour', ['pos', 'distance'])


class Cluster(set):
    def add_point(self, point, neighbour_mapping):
        self.add(point)
        for neighbour in neighbour_mapping[point]:
            if neighbour.pos not in self:
                self.add_point(
                    neighbour.pos, neighbour_mapping=neighbour_mapping
                )


def create_clusters(positions, *, feature_size, coordinate_system):
    positions = [tuple(pos) for pos in positions]
    neighbour_mapping = {pos: [] for pos in positions}
    for pos1, pos2 in itertools.combinations(positions, r=2):
        distance = coordinate_system.distance(np.array(pos1), np.array(pos2))
        if distance <= 2 * feature_size:
            neighbour_mapping[pos1].append(
                Neighbour(pos=pos2, distance=distance)
            )
            neighbour_mapping[pos2].append(
                Neighbour(pos=pos1, distance=distance)
            )

    clusters = []
    for pos in positions:
        if all(pos not in c for c in clusters):
            new_cluster = Cluster()
            new_cluster.add_point(pos, neighbour_mapping=neighbour_mapping)
            clusters.append(new_cluster)

    # check consistency
    for c1, c2 in itertools.combinations(clusters, r=2):
        assert not c1.intersection(c2), "Inconsistent neighbour mapping."

    return clusters, neighbour_mapping
