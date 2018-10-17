#!/usr/bin/env python

import matplotlib.pyplot as plt

import nodefinder as nf

if __name__ == '__main__':
    print('Line:')
    result = nf.identify.run(
        result=nf.io.load('line.hdf5'), evaluate_line_method='ballistic'
    )
    # result = nf.identify.run(result=nf.io.load('line.hdf5'), feature_size=2e-2)
    print(result)
    nf.io.save(result, 'result.hdf5')
    # fig, ax = nf.search.plot.points(nf.io.load('line.hdf5'))
    nf.identify.plot.result(result)

    plt.show()
    # plt.savefig('line.pdf')
