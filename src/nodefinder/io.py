# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines free functions to serialize / deserialize nodefinder objects to HDF5.

These are aliases for the functions defined in :mod:`fsc.hdf5_io`, meaning that
they can also handle other objects which are registered with the same system.
"""

from fsc.hdf5_io import save, load

__all__ = ["save", "load"]
