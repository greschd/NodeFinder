"""
Submodule for searching the nodal features given a gap function. The result is
given as a "point-cloud" of nodes.
"""

from ._run import *
from . import result
from . import plot

__all__ = _run.__all__ + ['result', 'plot']  # pylint: disable=undefined-variable
