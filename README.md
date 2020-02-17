# NodeFinder

The goal of this code is to identify nodal features in potential landscapes, using a two-step approach:

* **Search:** Starting from a grid of starting points, find an initial set of nodal points by using a minimization scheme (Nelder-Mead). The initial set of nodes can be further refined by searching in their vicinity. This should produce a point cloud which lies densely within the nodal features to be discovered.
* **Identification:** Identify the features (points, lines, surfaces) which are contained in the point cloud. For this step, the nodal points are first separated into clusters. For each cluster, the dimension is estimated, and depending on the points are evaluated further.

The initial use-case for this code is to find gapless features in three-dimensional materials.

Documentation: https://nodefinder.greschd.ch

[![Documentation Status](https://readthedocs.org/projects/nodefinder/badge/?version=latest)](https://nodefinder.greschd.ch/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/greschd/NodeFinder.svg?branch=master)](https://travis-ci.org/greschd/NodeFinder)
