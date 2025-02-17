# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Define the result classes for the identification step.
"""

# ruff: noqa: F403, F405

from ._containers import *
from ._shapes import *

__all__ = _containers.__all__ + _shapes.__all__
