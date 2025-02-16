#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import matplotlib.pyplot as plt

import nodefinder as nf


def gap_fct(pos):
    radius = 0.2
    dx, dy, dz = np.array(pos) - 0.5
    return (0.1 + 10 * dx**2) * np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2)


if __name__ == "__main__":
    result = nf.search.run(
        gap_fct,
        initial_mesh_size=3,
        gap_threshold=2e-4,
        feature_size=0.05,
        use_fake_potential=True,
    )
    nf.io.save(result, "result.hdf5")
    nf.search.plot.points(result)
    plt.show()
    # plt.savefig('nodal_line.pdf', bbox_inches='tight')
