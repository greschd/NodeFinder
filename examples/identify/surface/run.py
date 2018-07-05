#!/usr/bin/env python

import matplotlib.pyplot as plt

import nodefinder as nf

if __name__ == '__main__':
    print('Surface:')
    result = nf.identify.run(
        result=nf.io.load('surface.hdf5'), feature_size=5e-2
    )
    print(result)
    nf.identify.plot.result(result)
    plt.savefig('surface.pdf')
