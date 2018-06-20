from types import MappingProxyType

from fsc.export import export

from ._nelder_mead import root_nelder_mead


@export
async def run_minimization(
    func,
    initial_simplex,
    fake_potential=None,
    nelder_mead_kwargs=MappingProxyType({}),
):
    # print('run_minimization')
    if fake_potential is not None:
        raise NotImplementedError
    return await root_nelder_mead(
        func=func, initial_simplex=initial_simplex, **nelder_mead_kwargs
    )
