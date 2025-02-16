# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the function running the minimization, including the fake potential.
"""

from types import MappingProxyType

from fsc.export import export

from ..result._minimization import JoinedMinimizationResult
from ._nelder_mead import root_nelder_mead


def add_fake_potential(fake_pot, func):
    async def evaluate(x):
        return (await func(x)) + fake_pot(x)

    return evaluate


@export
async def run_minimization(
    func,
    *,
    initial_simplex,
    fake_potential=None,
    nelder_mead_kwargs=MappingProxyType({}),
):
    """Runs the minimization, including handling the fake potential.

    Runs the minimization for a given function and initial simplex. If a
    fake potential is given, the Nelder-Mead algorithm is first run with the
    fake potential, and then continued without the fake potential.
    The final simplex of the minimization with fake potential is enlarged and
    used as initial simplex for the second Nelder-Mead run.

    Arguments
    ---------
    func : collections.abc.Callable
        Function or coroutine describing the potential to be minimized.
    initial_simplex : numpy.ndarray
        Coordinates of the initial simplex.
    fake_potential : collections.abc.Callable
        Function describing the fake potential.
    nelder_mead_kwargs : collections.abc.Mapping
        Keyword arguments passed to the Nelder-Mead algorithm.
    """
    if fake_potential is not None:
        # TODO: Check if deepcopying fake potential is valid / better.
        # possible issue when not deep-copying is that the fake potential may
        # change during minimization, and the Nelder-Mead algorithm could get
        # horribly stuck when the current best value is within the 'infinite'
        # region.
        modified_kwargs = dict(nelder_mead_kwargs)
        modified_kwargs["ftol"] = float("inf")
        res_fake = await root_nelder_mead(
            func=add_fake_potential(fake_potential, func),
            initial_simplex=initial_simplex,
            **modified_kwargs,
        )
        simplex_final = res_fake.simplex_history[-1]
        simplex_blowup = simplex_final[0] + 1.5 * (simplex_final - simplex_final[0])

        res = await root_nelder_mead(
            func=func, initial_simplex=simplex_blowup, **nelder_mead_kwargs
        )

        return JoinedMinimizationResult(child=res, ancestor=res_fake)
    else:
        return await root_nelder_mead(
            func=func, initial_simplex=initial_simplex, **nelder_mead_kwargs
        )
