#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from kdotp import Hamilton, Pauli


def Hamilton_split(k, splitting):
    split_x, split_y, split_z = splitting
    return (
        Hamilton(k) + split_x * np.kron(Pauli.zero, Pauli.x) +
        split_y * np.kron(Pauli.zero, Pauli.y) +
        split_z * np.kron(Pauli.zero, Pauli.z)
    )
