# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""A tool to find and identify nodal features in band structures."""

import importlib.metadata

__version__ = importlib.metadata.version(__name__.replace(".", "-"))

from . import coordinate_system
from . import search
from . import identify
from . import io
from . import _logging  # noqa: F401

__all__ = ["search", "identify", "io", "coordinate_system"]
