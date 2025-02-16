# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the container classes for object shapes.
"""

from types import SimpleNamespace

import numpy as np
import networkx as nx

from fsc.export import export
from fsc.hdf5_io import SimpleHDF5Mapping, subscribe_hdf5, to_hdf5, from_hdf5


@export
@subscribe_hdf5("nodefinder.nodal_point")
class NodalPoint(SimpleNamespace, SimpleHDF5Mapping):
    """
    Shape class defining a nodal point.

    Attributes
    ----------
    position : tuple(float)
        The position of the point.
    """

    HDF5_ATTRIBUTES = ["position"]

    def __init__(self, position):
        self.position = position


@export
@subscribe_hdf5("nodefinder.nodal_line")
class NodalLine(SimpleNamespace):
    """
    Shape class defining a nodal line.

    Attributes
    ----------
    graph : networkx.Graph
        A graph describing the line.
    """

    def __init__(self, graph, degree_count):
        self.graph = graph
        self.degree_count = degree_count

    def __repr__(self):
        return "NodalLine(graph=<{} nodes, {} edges>, degree_count={}, shape_name='{}')".format(
            len(self.graph.nodes),
            len(self.graph.edges),
            self.degree_count,
            self.shape_name,
        )

    def to_hdf5(self, hdf5_handle):
        """
        Serialize the object and store in under the given HDF5 handle.
        """
        graph_group = hdf5_handle.create_group("graph")
        graph_group["nodes"] = np.array(list(self.graph.nodes))
        if self.graph.edges:
            graph_group["edges"] = np.array(list(self.graph.edges))

        degree_count_group = hdf5_handle.create_group("degree_count")
        to_hdf5(self.degree_count, degree_count_group)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        """
        Derialize the object from the given HDF5 handle.
        """
        graph_group = hdf5_handle["graph"]
        graph = nx.Graph()

        graph.add_nodes_from([tuple(n) for n in graph_group["nodes"]])
        if "edges" in graph_group:
            graph.add_edges_from(
                [(tuple(p1), tuple(p2)) for p1, p2 in graph_group["edges"]]
            )

        degree_count = from_hdf5(hdf5_handle["degree_count"])
        return cls(graph=graph, degree_count=degree_count)

    @property
    def shape_name(self):
        """
        Describes the shape of the line, as inferred from the degree count.
        """
        shape_lookup = {
            tuple(): "CLOSED LOOP",
            ((1, 2),): "OPEN LINE",
        }
        return shape_lookup.get(tuple(sorted(self.degree_count.items())), "UNKNOWN")
