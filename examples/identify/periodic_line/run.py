#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import matplotlib.pyplot as plt

import nodefinder as nf

if __name__ == "__main__":
    print("Line:")
    # result = nf.identify.run(result=nf.io.load('line.hdf5'))
    result = nf.identify.run(
        result=nf.io.load("line.hdf5"),
        feature_size=2e-2,
        evaluate_line_method="ballistic",
    )
    print(result)
    nf.io.save(result, "result.hdf5")
    nf.identify.plot.result(result)
    plt.show()
