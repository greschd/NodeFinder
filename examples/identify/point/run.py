#!/usr/bin/env python

import matplotlib.pyplot as plt

import nodefinder as nf

if __name__ == '__main__':
    print('Point:')
    result = nf.identify.run(
        result=nf.io.load('point.hdf5'), feature_size=2e-2
    )
    print(result)
    nf.identify.plot.result(result)
    plt.savefig('point.pdf')
