#!/usr/bin/env python
# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>

import nodefinder as nf
import matplotlib.pyplot as plt

# mesh_stencil = nf.search.refinement_stencil.get_mesh_stencil(mesh_size=(3, 3, 3))
# nf.search.plot.stencil(mesh_stencil)
# plt.show()

# sphere_stencil = nf.search.refinement_stencil.get_sphere_stencil(num_points=10)
# nf.search.plot.stencil(sphere_stencil)
# plt.show()

circle_stencil = nf.search.refinement_stencil.get_circle_stencil(num_points=10)
nf.search.plot.stencil(circle_stencil)
plt.show()
