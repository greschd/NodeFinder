import numpy as np


class FakePotential:
    def __init__(self, result, width, height):
        self.result = result
        self.width = width
        self.height = height

    def get_fake_pot(self, dist):
        return 2 * self.height * np.cos(
            min(np.pi / 2, (dist / self.width) * np.pi / 3)
        )

    def __call__(self, pos):
        distances = self.result.get_node_neighbour_distances(pos)
        return sum(self.get_fake_pot(dist) for dist in distances)
