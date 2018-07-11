# NodeFinder

The goal of this code is to identify nodal features in 3D materials, using a two-step approach:

* **Search:** Starting from a grid of starting points, find an initial set of nodal points by using a minimization scheme (Nelder-Mead). The initial set of nodes can be further refined by searching in their vicinity. This should produce a point cloud which lies densely within the nodal features to be discovered.
* **Identification:** Identify the features (points, lines, surfaces) which are contained in the point cloud. For this step, the nodal points are first separated into clusters. For each cluster, the dimension is estimated, and depending on the points are evaluated further.
