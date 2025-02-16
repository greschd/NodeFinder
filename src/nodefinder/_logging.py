# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the logger for the main module.
"""

import sys
import logging

MAIN_LOGGER = logging.getLogger("nodefinder")
DEFAULT_HANDLER = logging.StreamHandler(sys.stdout)
FORMATTER = logging.Formatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s",
)
DEFAULT_HANDLER.setFormatter(FORMATTER)
MAIN_LOGGER.addHandler(DEFAULT_HANDLER)
