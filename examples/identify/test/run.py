#!/usr/bin/env python

import nodefinder as nf

if __name__ == '__main__':
    nf.identify.run(result=nf.io.load('point.hdf5'))
    nf.identify.run(result=nf.io.load('line.hdf5'), feature_size=2e-2)
    nf.identify.run(result=nf.io.load('surface.hdf5'), feature_size=5e-2)
