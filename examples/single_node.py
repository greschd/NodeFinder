#!/usr/bin/env python

import numpy as np
import scipy.linalg as la
from nodefinder import run_node_finder


def gap_fct(pos):
    # print('.', end='', flush=True)
    return la.norm(np.array(pos) - 0.5)


if __name__ == '__main__':
    result = run_node_finder(gap_fct)
    for res in result.results:
        print('pos:', res.pos, 'success:', res.success, 'val:', res.value)
