#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import matplotlib.pyplot as plt

import nodefinder as nf

if __name__ == "__main__":
    print("Surface:")
    result = nf.identify.run(result=nf.io.load("surface.hdf5"), feature_size=5e-2)
    print(result)
    nf.io.save(result, "result.hdf5")
    nf.identify.plot.result(result)
    plt.savefig("surface.pdf")
