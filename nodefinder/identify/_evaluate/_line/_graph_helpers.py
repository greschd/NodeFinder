"""
Contains helper functions for dealing with graphs in the context of line
evaluation.
"""

from collections import Counter


def _create_degree_count(graph):
    degree_counter = Counter([val for pos, val in graph.degree])
    degree_counter.pop(2, None)
    return degree_counter
