#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import random

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

import nodefinder as nf


def gap_fct(pos, noise_level=0.1):
    noise = 1 + noise_level * random.random()
    return noise * la.norm(np.array(pos) - [0.2, 0.4, 0.8])


if __name__ == "__main__":
    result = nf.search.run(
        gap_fct,
        initial_mesh_size=(3, 3, 3),
    )
    nf.io.save(result, "result.hdf5")
    print("Found", len(result.nodes), "(non-unique) nodes.")
    nf.search.plot.points(result)
    plt.savefig("single_node.pdf", bbox_inches="tight")
