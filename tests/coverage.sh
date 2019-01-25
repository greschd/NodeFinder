#!/bin/bash
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

py.test -p no:cov-exclude --cov=nodefinder --cov-report=html
