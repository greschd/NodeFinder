# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the function which evaluates the shape of a point.
"""

import numpy as np

from ..result import NodalPoint


def _evaluate_point(positions, coordinate_system):
    return NodalPoint(
        position=coordinate_system.average([np.array(pos) for pos in positions])
    )
