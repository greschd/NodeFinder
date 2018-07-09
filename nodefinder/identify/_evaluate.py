"""
Defines the functions used to evaluate the shape of a given cluster of points.
"""

import warnings
import operator
from contextlib import suppress

import numpy as np
from scipy.sparse.csgraph import shortest_path

from fsc.export import export

from .result import NodalLine, NodalPoint


@export
def evaluate_cluster(
    positions, dim, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the shape of a cluster with the given positions.

    Arguments
    ---------
    positions : set(tuple(float))
        Positions of the points in the cluster.
    dim : int
        Dimension of the cluster.
    coordinate_system : CoordinateSystem
        Coordinate system used to calculate distances.
    neighbour_mapping : dict
        Mapping containing a list of neighbours for each position.
    feature_size : float
        TODO

    Returns
    -------
    :obj:`None` or NodalPoint or NodalLine :
        The shape of the given positions. Returns ``None`` if the shape could
        not be determined.
    """
    if dim == 0:
        return _evaluate_point(
            positions=positions, coordinate_system=coordinate_system
        )
    elif dim == 1:
        try:
            return _evaluate_line(
                positions=positions,
                coordinate_system=coordinate_system,
                neighbour_mapping=neighbour_mapping,
                feature_size=feature_size
            )
        except (IndexError, ValueError) as exc:
            warnings.warn('Could not identify line: {}'.format(exc))


def _evaluate_point(positions, coordinate_system):
    return NodalPoint(
        position=coordinate_system.
        average([np.array(pos) for pos in positions])
    )


def _evaluate_line(
    positions, coordinate_system, neighbour_mapping, feature_size
):
    """
    Evaluate the positions of a closed line.
    """
    pos1 = positions.pop()
    pos2, distance = max(((
        pos_candidate,
        coordinate_system.distance(np.array(pos1), np.array(pos_candidate))
    ) for pos_candidate in positions),
                         key=operator.itemgetter(1))

    if distance <= 2 * feature_size:
        raise ValueError('No suitable second position found.')
    positions.remove(pos2)

    positions_inner = list(positions)
    positions_list = [pos1] + positions_inner + [pos2]
    index_mapping = {pos: i for i, pos in enumerate(positions_list)}

    path1 = _get_shortest_path(
        positions=positions_list,
        index_mapping=index_mapping,
        neighbour_mapping=neighbour_mapping
    )

    edge_neighbours = set()
    edge_neighbours.update([n.pos for n in neighbour_mapping[pos1]])
    edge_neighbours.update([n.pos for n in neighbour_mapping[pos2]])
    path_neighbours = set(path1)
    for pos in path1:
        path_neighbours.update([n.pos for n in neighbour_mapping[pos]])
    positions_to_exclude = path_neighbours - edge_neighbours

    positions_list_partial = [
        pos1
    ] + list(set(positions) - positions_to_exclude) + [pos2]
    index_mapping_partial = {
        pos: i
        for i, pos in enumerate(positions_list_partial)
    }
    path2 = _get_shortest_path(
        positions=positions_list_partial,
        index_mapping=index_mapping_partial,
        neighbour_mapping=neighbour_mapping
    )
    path_full = path1 + list(reversed(path2))[1:]
    return NodalLine(path=path_full)


def _get_shortest_path(
    *,
    positions,
    index_mapping,
    neighbour_mapping,
    weight_func=lambda dist: 1 / dist
):
    """
    Get the shortest path between the start and end of a list of positions.
    """
    num_pos = len(positions)
    weight_array = np.zeros(shape=(num_pos, num_pos))
    for pos1, neighbours in neighbour_mapping.items():
        for pos2, distance in neighbours:
            with suppress(KeyError):
                weight = weight_func(distance)
                weight_array[index_mapping[pos1], index_mapping[pos2]] = weight

    _, predecessors = shortest_path(
        weight_array, directed=False, return_predecessors=True
    )
    start_idx = 0
    end_idx = num_pos - 1
    path = [end_idx]
    current = predecessors[start_idx, end_idx]
    while True:
        path.append(current)
        if current == start_idx:
            break
        current = predecessors[start_idx, current]
    return [positions[idx] for idx in path]
