"""
Contains functions for creating the refinement stencil used by the search
procedure.
"""

import numpy as np

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
        _generate_mesh_simplices(mesh_size=mesh_size, limits=limits)
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
    return get_mesh_stencil(mesh_size=[3] * dim)
