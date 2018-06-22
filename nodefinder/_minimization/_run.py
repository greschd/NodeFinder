from types import MappingProxyType

from fsc.export import export

from ._result import JoinedResult
from ._nelder_mead import root_nelder_mead


def add_fake_potential(fake_pot, func):
    async def evaluate(x):
        return (await func(x)) + fake_pot(x)

    return evaluate


@export
async def run_minimization(
    func,
    initial_simplex,
    fake_potential=None,
    nelder_mead_kwargs=MappingProxyType({}),
):
    if fake_potential is not None:
        res_fake = await root_nelder_mead(
            func=add_fake_potential(fake_potential, func),
            initial_simplex=initial_simplex,
            **nelder_mead_kwargs
        )
        simplex_final = res_fake.simplex_history[-1]
        simplex_blowup = simplex_final[0] + 10 * (
            simplex_final - simplex_final[0]
        )
        res = await root_nelder_mead(
            func=func, initial_simplex=simplex_blowup, **nelder_mead_kwargs
        )
        return JoinedResult(child=res, ancestor=res_fake)
    else:
        return await root_nelder_mead(
            func=func, initial_simplex=initial_simplex, **nelder_mead_kwargs
        )
