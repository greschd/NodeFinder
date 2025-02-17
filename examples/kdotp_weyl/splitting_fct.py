#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

# ruff: noqa: E741

import json

import math
import cmath
import numpy as np

with open("data/fit.json", "r") as f:
    FIT = json.load(f)


def c00(k, a, j, m, o, q, **kwargs):
    kx, ky, kz = k
    return a + j * (kx**2 + ky**2) + m * kx * ky + o * (kx * kz - ky * kz) + q * kz**2


def cxz(k, c, **kwargs):
    kx, ky, kz = k
    return c * (kx + ky)


def cy0(k, d, **kwargs):
    kx, ky, kz = k
    return d * (kx + ky)


def cz0(k, b, l, n, p, r, **kwargs):
    kx, ky, kz = k
    return b + l * (kx**2 + ky**2) + n * kx * ky + p * (kx * kz - ky * kz) + r * kz**2


def cxx(k, e, g, **kwargs):
    kx, ky, kz = k
    return e * (kx - ky) + g * kz


def cxy(k, f, h, **kwargs):
    kx, ky, kz = k
    return f * (kx - ky) + h * kz


def gap(*, c00, cxz, cy0, cz0, cxx, cxy, splitting):
    Bx, By, Bz = splitting
    A = (
        2 * By * Bz * cxy * cxz
        + 2 * Bx * cxx * (By * cxy + Bz * cxz)
        + Bx**2 * (cxx**2 + cy0**2 + cz0**2)
        + By**2 * (cxy**2 + cy0**2 + cz0**2)
        + Bz**2 * (cxz**2 + cy0**2 + cz0**2)
    )
    try:
        B = 2 * math.sqrt(A)
    except ValueError:
        B = 2 * cmath.sqrt(A)
    C = Bx**2 + By**2 + Bz**2 + cxx**2 + cxy**2 + cxz**2 + cy0**2 + cz0**2

    try:
        eigs = [
            c00 - math.sqrt(C + B),
            c00 - math.sqrt(C - B),
            c00 + math.sqrt(C + B),
            c00 + math.sqrt(C - B),
        ]
        eigs = np.sort(eigs)
    except ValueError:
        eigs = [
            c00 - cmath.sqrt(C + B),
            c00 - cmath.sqrt(C - B),
            c00 + cmath.sqrt(C + B),
            c00 + cmath.sqrt(C - B),
        ]
        assert np.isclose(np.imag(eigs), [0] * 4, atol=1e-6).all()
        eigs = np.sort(np.real(eigs))

    return eigs[2] - eigs[1]


def gap_fct(k, splitting):
    return gap(
        c00=c00(k, **FIT),
        cxz=cxz(k, **FIT),
        cy0=cy0(k, **FIT),
        cz0=cz0(k, **FIT),
        cxx=cxx(k, **FIT),
        cxy=cxy(k, **FIT),
        splitting=splitting,
    )
