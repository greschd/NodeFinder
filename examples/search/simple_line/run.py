#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import nodefinder as nf


def gap_fct(pos):
    radius = 0.2
    dx, dy, dz = np.array(pos) - 0.5
    return ((0.1 + 50 * dx**2) *
            np.sqrt(np.abs(dx**2 + dy**2 - radius**2) + dz**2))


if __name__ == '__main__':

    result = nf.search.run(
        gap_fct,
        initial_mesh_size=(3, 3, 3),
        gap_threshold=2e-4,
        feature_size=2e-2,
        use_fake_potential=True,
    )
    nf.io.save(result, 'result.hdf5')
    nf.plot.points_3d(result)
    plt.show()
    # plt.savefig('nodal_line.pdf', bbox_inches='tight')
