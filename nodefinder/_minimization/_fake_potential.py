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
        neighbours = self.result.get_node_neighbours(pos)
        distances = [
            self.result.coordinate_system.distance(pos, nb.pos)
            for nb in neighbours
        ]
        return sum(self.get_fake_pot(dist) for dist in distances)
