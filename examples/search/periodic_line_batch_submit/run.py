#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import numpy as np
import matplotlib.pyplot as plt
import nodefinder as nf

from fsc.async_tools import BatchSubmitter


def gap_fct(pos_list):
    radius = 0.2
    # print(len(pos_list))
    delta = (np.array(pos_list) - np.array([0, 0.5, 0.5])) % 1
    delta = np.minimum(delta, 1 - delta)
    return np.sqrt(
        np.abs(delta[:, 0]**2 + delta[:, 1]**2 - radius**2) + delta[:, 2]**2
    )


gap_fct_batched = BatchSubmitter(gap_fct, timeout=0, max_batch_size=100)

if __name__ == "__main__":

    result = nf.search.run(
        gap_fct_batched,
        initial_mesh_size=(3, 3, 3),
        gap_threshold=2e-4,
        feature_size=2e-2,
        use_fake_potential=True,
        num_minimize_parallel=100,
    )
    nf.io.save(result, "result.hdf5")
    nf.search.plot.points(result)
    plt.show()
    plt.savefig("line.pdf", bbox_inches="tight")
