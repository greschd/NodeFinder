#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# The additional license terms given in ADDITIONAL_TERMS.txt apply to this file.

__all__ = ['root_nelder_mead']

import numpy as np

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.'}


class OptimizeResult(dict):
    """ Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

class OptimizeWarning(UserWarning):
    pass

def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

def root_nelder_mead(func, x0, args=(), callback=None,
                         xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                         disp=False, return_all=False):
    """
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter : int
        Maximum number of iterations to perform.
    maxfev : int
        Maximum number of function evaluations to make.

    """
    maxfun = maxfev
    retall = return_all

    fcalls, func = wrap_function(func, args)
    x0 = np.asfarray(x0).flatten()
    N = len(x0)
    if maxiter is None:
        maxiter = N * 200
    if maxfun is None:
        maxfun = N * 200

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    one2np1 = list(range(1, N + 1))

    sim = np.zeros((N + 1, N), dtype=x0.dtype)
    fsim = np.zeros((N + 1,), float)
    sim[0] = x0
    if retall:
        allvecs = [sim[0]]
    fsim[0] = func(x0)
    nonzdelt = 0.05
    zdelt = 0.00025
    for k in range(0, N):
        y = np.array(x0, copy=True)
        if y[k] != 0:
            y[k] = (1 + nonzdelt)*y[k]
        else:
            y[k] = zdelt

        sim[k + 1] = y
        f = func(y)
        fsim[k + 1] = f

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xtol and
                np.max(np.abs(fsim[0] - fsim[1:])) <= ftol):
            break

        xbar = np.add.reduce(sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

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
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
        if disp:
            print('Warning: ' + msg)
    elif iterations >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
        if disp:
            print('Warning: ' + msg)
    else:
        msg = _status_message['success']
        if disp:
            print(msg)
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result
