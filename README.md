# NodeFinder

The goal of this code is to identify nodal features in 3D materials, using the following 3 steps:

* **Minimization:** Starting from a grid of starting points, find an initial set of nodal points by using a minimization scheme (Nelder-Mead).
* **Refinement:** Search in the vicinity of the initial nodal points, such that the final point cloud of nodal points lies densely within the features which are to be discovered.
* **Identification:** Identify the features (points, lines, surfaces) which are contained in the point cloud.
