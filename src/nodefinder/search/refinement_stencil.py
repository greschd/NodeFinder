# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Contains functions for creating the refinement stencil used by the search
procedure.
"""

import numpy as np
import scipy.linalg as la

from fsc.export import export

from ._mesh_helper import _generate_mesh_simplices


@export
def get_mesh_stencil(*, mesh_size, dist_multiplier=2.5):
    """
    Create a mesh refinement stencil.

    Arguments
    ---------
    mesh_size : list(int)
        The size of the mesh in each dimension.
    dist_multiplier : float
        Defines the size of the refinement mesh. A multiplier of one means that
        the refinement box extends to the ``dist_cutoff`` on each side.
    """
    limits = [(-dist_multiplier, dist_multiplier)] * len(mesh_size)
    return np.array(
        _generate_mesh_simplices(mesh_size=mesh_size, limits=limits, skip_origin=True)
    )


@export
def get_auto_stencil(*, dim):
    """
    Get the default stencil for a given dimension.

    Arguments
    ---------
    dim : int
        The problem dimension.
    """
    if dim == 2:
        return get_circle_stencil(num_points=5)
    elif dim == 3:
        return get_sphere_stencil(num_points=30)
    return get_mesh_stencil(mesh_size=[3] * dim)


def get_circle_stencil(*, num_points):
    """
    Produce a stencil with simplices along a circle. Only suitable for
    two-dimensional problems.

    Arguments
    ---------
    num_points : int
        The number of points on the circle / number of simplices.
    """
    phi = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    simplex = (
        np.array([[0, 0], [1 / 2, np.sqrt(3) / 2], [-1 / 2, np.sqrt(3) / 2]])
        * 5
        / num_points
    ) + [0, 1.5]
    res = np.zeros((num_points, 3, 2))
    for i, phi_val in enumerate(phi):
        res[i, :, :] = (
            np.array(
                [
                    [np.cos(phi_val), -np.sin(phi_val)],
                    [np.sin(phi_val), np.cos(phi_val)],
                ]
            )
            @ simplex.T
        ).T
    return res


def get_sphere_stencil(*, num_points):
    """
    Produce a stencil with simplices on the surface of a sphere. Only suitable
    for three-dimensional problems.

    Arguments
    ---------
    num_points : int
        The number of simplices which are placed on the sphere.
    """
    points = 1.1 * np.array(_fibonacci_sphere_points(num_points))
    simplex_edge_length = 3 / np.sqrt(num_points)
    simplex = np.zeros((4, 3))
    simplex[1:, :] = (0.25 + 0.75 * np.eye(3)) * simplex_edge_length
    q_mat_1, r_mat_1 = la.qr([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    q_mat_1 *= np.sign(r_mat_1[0, 0])
    simplex = (q_mat_1 @ simplex.T).T

    res = []
    for pos in points:
        mat = np.zeros((3, 3))
        mat[:, 0] = pos
        q_mat_2, r_mat_2 = la.qr(mat)
        q_mat_2 *= np.sign(r_mat_2[0, 0])
        res.append((q_mat_2 @ simplex.T).T + pos)

    return np.array(res)


def _fibonacci_sphere_points(num_points):
    """
    Helper function that places points on a sphere using the Fibonacci spiral.
    """
    res = []
    offset = 2 / num_points
    increment = np.pi * (3 - np.sqrt(5))

    for i in range(num_points):
        z = (i + 0.5) * offset - 1
        rho = np.sqrt(1 - z**2)
        phi = ((i + 1) % num_points) * increment

        x = np.cos(phi) * rho
        y = np.sin(phi) * rho

        res.append([x, y, z])
    return res
