.. © 2017-2019, ETH Zurich, Institut für Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>

.. _tutorial:

Tutorial
========

In this tutorial, we will show the use of NodeFinder with a simple example. We will find the nodal line given by the following function:

.. ipython::

    In [0]: import numpy as np

    In [0]: def gap_function(pos):
       ...:     x, y, z = pos
       ...:     return np.sqrt((y - np.sin(np.pi * x))**2 + z**2)

Search
------

The NodeFinder module consists of two parts: In the first part, a Nelder-Mead optimization is performed from many starting points. If nodes are found, a local refinement is done to check if there are more nodal points nearby, such as when there is a nodal line. The goal of this first step is to find nodal positions which lie densely in the object which should be identified.

This first step is performed using the :mod:`.search` submodule:

.. ipython::

    In [0]: import nodefinder as nf

    In [0]: search_result = nf.search.run(
       ...:     gap_function,
       ...:     limits=[(-1, 1)] * 3,
       ...:     initial_mesh_size=3,
       ...:     gap_threshold=1e-4,
       ...:     feature_size=0.1,
       ...:     use_fake_potential=True,
       ...: )

    In [0]: print(search_result)

    @savefig search_plot.png
    In [0]: nf.search.plot.points(search_result);

The main logic is in the :func:`.search.run` method. As you can see, this function can take some additional keyword arguments, which are described in the :func:`reference section <.search.run>`.

Of note here is the ``use_fake_potential`` flag. When this flag is set, each minimization step first adds a "fake" potential to the function to be minimized, which repels the minimization from nodes which have already been found. This is especially useful for nodal lines, because it helps covering the whole line.

Identify
--------

The second step in the process is to identify the nodal features from the point clouds calculated in the search step. For this purpose, the :mod:`.identify` submodule is used.

This process works by first clustering the points into connected components. Next, the dimension of each cluster is determined. Finally, the shape of the object is determined.

.. ipython::

    In [0]: identify_result = nf.identify.run(search_result)

    In [0]: print(identify_result)

    @savefig identify_plot.png
    In [0]: nf.identify.plot.result(identify_result);
