#!/usr/bin/env python

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
