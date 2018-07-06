class FakePotential:
    def __init__(self, result, width):
        self.result = result
        self.width = width

    def get_fake_pot(self, distances):
        if any(dist < self.width for dist in distances):
            return float('inf')
        return 0

    def __call__(self, pos):
        try:
            return self.get_fake_pot(
                self.result.get_all_neighbour_distances(pos)
            )
        except ValueError:
            return 0
