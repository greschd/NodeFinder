#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nodefinder as nf

if __name__ == '__main__':
    print('Point:')
    print(nf.identify.run(result=nf.io.load('point.hdf5')))
    print('Line:')
    line_search_result = nf.io.load('line.hdf5')
    line_identify_result = nf.identify.run(
        result=line_search_result, feature_size=2e-2
    )
    print(line_identify_result)
    path = line_identify_result.results[0].result.path
    # fig = plt.figure()
    # axis = fig.add_subplot(111, projection='3d')
    fig, ax = nf.plot.points_3d(line_search_result)
    ax.plot(*np.array(path).T, color='C1')
    plt.savefig('line.pdf')

    print('Surface')
    print(
        nf.identify.run(result=nf.io.load('surface.hdf5'), feature_size=5e-2)
    )
