class FakePotential:
    def __init__(self, result, width, height):
        self.result = result
        self.width = width
        self.height = height

    def get_fake_pot(self, distances):
        return sum(
            min(0, self.height * (2 - dist / self.width)) for dist in distances
        )

    def __call__(self, pos):
        try:
            return self.get_fake_pot(
                self.result.get_all_neighbour_distances(pos)
            )
        except ValueError:
            return 0
