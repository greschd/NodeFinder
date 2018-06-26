#!/usr/bin/env python

import random

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from nodefinder import run_node_finder, plot


def gap_fct(pos, noise_level=0.1):
    noise = (1 + noise_level * random.random())
    return noise * la.norm(np.array(pos) - [0.2, 0.4, 0.8])


if __name__ == '__main__':
    result = run_node_finder(
        gap_fct,
        initial_mesh_size=(3, 3, 3),
    )
    print('Found', len(result.nodes), '(non-unique) nodes.')
    plot.plot_3d(result)
    plt.savefig('single_node.pdf', bbox_inches='tight')
