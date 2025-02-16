#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import matplotlib.pyplot as plt

import nodefinder as nf


def gap_fct(pos):
    radius = 0.2
    delta = (np.array(pos) - [0, 0.5, 0.5]) % 1
    delta = np.minimum(delta, 1 - delta)
    dx, dy, dz = delta
    return np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2)


if __name__ == "__main__":
    result = nf.search.run(
        gap_fct,
        initial_mesh_size=(3, 3, 3),
        gap_threshold=2e-4,
        feature_size=2e-2,
        use_fake_potential=True,
    )
    nf.io.save(result, "result.hdf5")
    nf.search.plot.points(result)
    plt.show()
