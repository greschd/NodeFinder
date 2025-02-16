#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import matplotlib.pyplot as plt
import nodefinder as nf


def gap_fct(pos):
    dx, dy, dz = (np.array(pos) % 1) - 0.5
    return abs(dz) * (0.1 + 10 * (dx**2 + dy**2))


if __name__ == "__main__":
    result = nf.search.run(
        gap_fct,
        initial_mesh_size=(1, 1, 1),
        refinement_stencil=nf.search.refinement_stencil.get_mesh_stencil(
            mesh_size=(2, 2, 2)
        ),
        gap_threshold=1e-4,
        feature_size=5e-2,
        use_fake_potential=False,
    )
    nf.io.save(result, "result.hdf5")
    nf.search.plot.points(result)
    plt.show()
    # plt.savefig('nodal_line.pdf', bbox_inches='tight')
