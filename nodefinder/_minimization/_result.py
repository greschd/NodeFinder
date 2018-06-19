# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# The additional license terms given in ADDITIONAL_TERMS.txt apply to this
# file.

from types import SimpleNamespace

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, HDF5Enabled

@export
@subscribe_hdf5('nodefinder.optimize_result')
class MinimizationResult(SimpleNamespace, HDF5Enabled):
    """ Represents the optimization result.

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
    simplex_history : ndarray
        History of simplex values.
    fun_simplex_history : ndarray
        History of function values of the simplex.
    """

    def to_hdf5(self, hdf5_handle):
        for key, val in self.__dict__.items():
            hdf5_handle[key] = val


    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(**{key: val.value for key, val in hdf5_handle.items()})
