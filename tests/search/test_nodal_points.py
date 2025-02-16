# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Tests with a single nodal point.
"""

# pylint: disable=redefined-outer-name

import tempfile

import pytest
import numpy as np
import scipy.linalg as la

import nodefinder as nf
from nodefinder.search import run

NODE_PARAMETERS = pytest.mark.parametrize(
    "node_positions, mesh_size",
    [
        ([(0.5, 0.5, 0.5)], (1, 2, 1)),
        ([(0.2, 0.9, 0.6), (0.99, 0.01, 0.0), (0.7, 0.2, 0.8)], (3, 3, 3)),
    ],
)


@pytest.fixture
def gap_fct(node_positions):
    """
    Calculates the minimum distance between a given position and the nearest node.
    """
    node_pos_array = np.array(node_positions)

    def inner(x):
        deltas = (np.array(x) - node_pos_array) % 1
        deltas_periodic = np.minimum(deltas, 1 - deltas)  # pylint: disable=assignment-from-no-return,useless-suppression
        distances = la.norm(deltas_periodic, axis=-1)
        return np.min(distances)

    return inner


@pytest.fixture(params=[False, True])
def use_fake_potential(request):
    return request.param


@NODE_PARAMETERS
def test_simple(
    gap_fct,
    node_positions,
    mesh_size,
    use_fake_potential,
    score_nodal_points,
):
    """
    Test that a single nodal point is found.
    """
    result = run(
        gap_fct=gap_fct,
        initial_mesh_size=mesh_size,
        use_fake_potential=use_fake_potential,
    )
    score_nodal_points(
        result, exact_points=node_positions, cutoff_accuracy=1e-6, cutoff_coverage=1e-6
    )


@NODE_PARAMETERS
def test_no_history(
    gap_fct,
    node_positions,
    mesh_size,
    use_fake_potential,
    score_nodal_points,
):
    """
    Test that a single nodal point is found.
    """
    result = run(
        gap_fct=gap_fct,
        initial_mesh_size=mesh_size,
        use_fake_potential=use_fake_potential,
        nelder_mead_kwargs={"keep_history": False},
    )
    score_nodal_points(
        result, exact_points=node_positions, cutoff_accuracy=1e-6, cutoff_coverage=1e-6
    )


@NODE_PARAMETERS
def test_save(
    gap_fct, node_positions, mesh_size, use_fake_potential, score_nodal_points
):
    """
    Test saving to a file
    """
    with tempfile.NamedTemporaryFile() as named_file:
        result = run(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=mesh_size,
            use_fake_potential=use_fake_potential,
        )
        score_nodal_points(
            result,
            exact_points=node_positions,
            cutoff_accuracy=1e-6,
            cutoff_coverage=1e-6,
        )


@NODE_PARAMETERS
def test_restart(
    gap_fct, node_positions, mesh_size, score_nodal_points, use_fake_potential
):
    """
    Test that the calculation is done when restarting from a finished result.
    """

    def invalid_gap_fct(x):
        raise ValueError

    with tempfile.NamedTemporaryFile() as named_file:
        result = run(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=mesh_size,
            use_fake_potential=use_fake_potential,
        )
        score_nodal_points(
            result,
            exact_points=node_positions,
            cutoff_accuracy=1e-6,
            cutoff_coverage=1e-6,
            additional_tag="initial_",
        )

        restart_result = run(
            gap_fct=invalid_gap_fct,
            save_file=named_file.name,
            load=True,
            load_quiet=False,
            initial_mesh_size=mesh_size,
            use_fake_potential=use_fake_potential,
        )
        score_nodal_points(
            restart_result,
            exact_points=node_positions,
            cutoff_accuracy=1e-6,
            cutoff_coverage=1e-6,
            additional_tag="restart_",
        )


@NODE_PARAMETERS
def test_restart_partial(gap_fct, node_positions, mesh_size):
    """
    Test that no additional refinement is done when restarting with a forced
    initial mesh.
    """
    with tempfile.NamedTemporaryFile() as named_file:
        refinement_stencil = nf.search.refinement_stencil.get_mesh_stencil(
            mesh_size=[1, 1, 1]
        )
        result = run(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=mesh_size,
            use_fake_potential=False,
            refinement_stencil=refinement_stencil,
        )
        # number of starting points + one refinement per node
        assert len(result.nodes) == np.prod(mesh_size) + len(node_positions) * len(
            refinement_stencil
        )
        result2 = run(
            gap_fct=gap_fct,
            save_file=named_file.name,
            initial_mesh_size=mesh_size,
            load=True,
            use_fake_potential=False,
            refinement_stencil=refinement_stencil,
            force_initial_mesh=True,
        )
        assert len(result2.nodes) == 2 * np.prod(mesh_size) + len(node_positions) * len(
            refinement_stencil
        )


def test_raises():
    """
    Test that using an invalid gap_fct raises the error.
    """

    async def gap_fct(pos):
        raise ValueError("test error.")

    with pytest.raises(ValueError):
        run(gap_fct)
