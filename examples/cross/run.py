#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
import nodefinder as nf
import matplotlib.pyplot as plt


def gap_func(pos):
    x, y = pos
    return abs(x * y)


if __name__ == "__main__":
    try:
        search_result = nf.io.load("search_result.hdf5")
    except IOError:
        search_result = nf.search.run(
            gap_func,
            limits=[(-1, 1)] * 2,
            initial_mesh_size=3,
            feature_size=0.1,
            gap_threshold=1e-4,
            use_fake_potential=False,
            periodic=True,
        )
        nf.io.save(search_result, "search_result.hdf5")
    try:
        identify_result = nf.io.load("identify_result.hdf5")
    except IOError:
        identify_result = nf.identify.run(search_result)
        nf.io.save(identify_result, "identify_result.hdf5")

    print(identify_result)
    nf.identify.plot.result(identify_result)
    plt.show()
