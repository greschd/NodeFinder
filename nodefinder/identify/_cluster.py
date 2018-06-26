import itertools

class Cluster(set):
    def add_point(self, point, neighbour_mapping):
        self.add(point)
        for neighbour in neighbour_mapping[point]:
            if neighbour not in self:
                self.add_point(neighbour, neighbour_mapping=neighbour_mapping)

def create_clusters(positions, *, feature_size, coordinate_system):
    neighbour_mapping = {pos: [] for pos in positions}
    for p1, p2 in itertools.combinations(positions, r=2):
        if coordinate_system.distance(p1, p2) <= 2 * feature_size:
            neighbour_mapping[p1].append(p2)
            neighbour_mapping[p2].append(p1)

    clusters = []
    for pos in positions:
        if all(pos not in cl for cl in clusters):
            new_cluster = Cluster()
            new_cluster.add_point(pos, neighbour_mapping=neighbour_mapping)
            clusters.append(new_cluster)

    # check consistency
    for c1, c2 in itertools.combinations(clusters, r=2):
        assert not c1.intersection(c2), "Inconsistent neighbour mapping."

    return clusters
