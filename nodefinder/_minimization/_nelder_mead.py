# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# The additional license terms given in ADDITIONAL_TERMS.txt apply to this
# file.

# pylint: skip-file

import asyncio

import numpy as np
from fsc.export import export

from ._result import MinimizationResult

# standard status messages of optimizers
_status_message = {
    'success': 'Optimization terminated successfully.',
    'maxfev': 'Maximum number of function evaluations has '
    'been exceeded.',
    'maxiter': 'Maximum number of iterations has been '
    'exceeded.',
    'pr_loss': 'Desired error not necessarily achieved due '
    'to precision loss.'
}


def wrap_function(function):
    ncalls = [0]

    async def function_wrapper(*args):
        ncalls[0] += 1
        return await function(*args)

    return ncalls, function_wrapper


@export
async def root_nelder_mead(
    func,
    initial_simplex,
    xtol=1e-4,
    ftol=1e-4,
    maxiter=None,
    maxfev=None,
):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Arguments
    ---------
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    maxfev : int
        Maximum number of function evaluations to make.

    Returns
    -------
    OptimizeResult:
        The result of the optimization.
    """
    maxfun = maxfev

    fcalls, func = wrap_function(func)
    N = len(initial_simplex[0])
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    sim = np.array(initial_simplex)
    assert sim.shape == (N + 1, N)

    fsim = np.array(await asyncio.gather(*[func(x) for x in sim]), dtype=float)
    assert fsim.shape == (N + 1, )

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    simplex_history = [np.copy(sim)]
    fun_simplex_history = [np.copy(fsim)]

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xtol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= ftol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = await func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = await func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = await func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = await func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = await func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        iterations += 1
        simplex_history.append(np.copy(sim))
        fun_simplex_history.append(np.copy(fsim))

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    result = MinimizationResult(
        pos=x,
        value=fval,
        num_iter=iterations,
        num_fev=fcalls[0],
        status=warnflag,
        success=(warnflag == 0),
        message=msg,
        simplex_history=np.array(simplex_history),
        fun_simplex_history=np.array(fun_simplex_history)
    )
    return result
