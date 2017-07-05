#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as la

from nodefinder import NodeFinder

def test_single_node():
    def gap_fct(x):
        return la.norm(np.array(x) - [0.5] * 3)

    nf = NodeFinder(gap_fct=gap_fct)
    print(nf._nodal_points)
