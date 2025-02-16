# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# The additional license terms given in ADDITIONAL_TERMS.txt apply to this
# file.

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the result classes for the minimization.
"""

from types import SimpleNamespace

import numpy as np
from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping, HDF5Enabled


@export
@subscribe_hdf5(
    "nodefinder.joined_minimization_result", extra_tags=["nodefinder.joined_result"]
)
class JoinedMinimizationResult(SimpleHDF5Mapping):
    """
    Wrapper for minimization results that result from two steps.

    Attributes
    ----------
    ancestor : MinimizationResult
        Result of the first minimization run.
    child : MinimizationResult
        Result of the second minimization run.
    """

    JOIN_KEYS = ["num_fev", "num_iter", "simplex_history", "fun_simplex_history"]
    HDF5_ATTRIBUTES = ["ancestor", "child"]

    def __init__(self, *, child, ancestor):
        self.child = child
        self.ancestor = ancestor

    def __getattr__(self, key):
        """
        Joins child and ancestor values where that makes sense, and returns the
        child value otherwise.
        """
        if key in self.JOIN_KEYS:
            return self._join(getattr(self.ancestor, key), getattr(self.child, key))
        else:
            return getattr(self.child, key)

    @staticmethod
    def _join(obj1, obj2):
        """
        Helper function to join two result values.
        """
        if isinstance(obj1, np.ndarray):
            return np.concatenate([obj1, obj2])
        else:
            return obj1 + obj2


@export
@subscribe_hdf5("nodefinder.minimization_result")
class MinimizationResult(SimpleNamespace, HDF5Enabled):
    """Represents the optimization result.

    Attributes
    ----------
    pos : ndarray
        The solution of the optimization.
    value : ndarray
        Value of objective function.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    num_fev : int
        Number of evaluations of the objective functions.
    num_iter : int
        Number of iterations performed by the optimizer.
    simplex_history : ndarray, optional
        History of simplex values.
    fun_simplex_history : ndarray, optional
        History of function values of the simplex.
    """

    def to_hdf5(self, hdf5_handle):
        for key, val in self.__dict__.items():
            assert key != "type_tag"
            hdf5_handle[key] = val

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            **{key: val[()] for key, val in hdf5_handle.items() if key != "type_tag"}
        )
