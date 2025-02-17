#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

# ruff: noqa: E741

import os
import json
import functools

import numpy as np


class Pauli:
    zero = np.eye(2, dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1, -1])


def Hamilton_general(k, *, a, b, c, d, e, f, g, h, j, l, m, n, o, p, q, r):
    kx, ky, kz = k
    H = (
        a * np.kron(Pauli.zero, Pauli.zero)
        + b * np.kron(Pauli.z, Pauli.zero)
        + c * (kx + ky) * np.kron(Pauli.x, Pauli.z)
        + d * (kx + ky) * np.kron(Pauli.y, Pauli.zero)
        + e * (kx - ky) * np.kron(Pauli.x, Pauli.x)
        + f * (kx - ky) * np.kron(Pauli.x, Pauli.y)
        + g * kz * np.kron(Pauli.x, Pauli.x)
        + h * kz * np.kron(Pauli.x, Pauli.y)
        + j * (kx**2 + ky**2) * np.kron(Pauli.zero, Pauli.zero)
        + l * (kx**2 + ky**2) * np.kron(Pauli.z, Pauli.zero)
        + m * kx * ky * np.kron(Pauli.zero, Pauli.zero)
        + n * kx * ky * np.kron(Pauli.z, Pauli.zero)
        + o * (kx * kz - ky * kz) * np.kron(Pauli.zero, Pauli.zero)
        + p * (kx * kz - ky * kz) * np.kron(Pauli.z, Pauli.zero)
        + q * kz**2 * np.kron(Pauli.zero, Pauli.zero)
        + r * kz**2 * np.kron(Pauli.z, Pauli.zero)
    )
    return H


with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/fit.json"), "r"
) as f:
    fit = json.load(f)

Hamilton = functools.partial(Hamilton_general, **fit)
