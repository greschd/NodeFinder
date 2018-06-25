class FakePotential:
    def __init__(self, result, width, height):
        self.result = result
        self.width = width
        self.height = height

    def get_fake_pot(self, dist):
        return min(0, self.height * (2 - dist / self.width))

    def __call__(self, pos):
        try:
            min_distance = min(self.result.get_all_neighbour_distances(pos))
            return self.get_fake_pot(min_distance)
        except ValueError:
            return 0
