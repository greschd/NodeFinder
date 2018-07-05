from types import SimpleNamespace
from contextlib import suppress

import numpy as np
from scipy.sparse.csgraph import shortest_path


def evaluate_cluster(
    positions, dim, coordinate_system, neighbour_mapping, feature_size
):
    if dim == 0:
        return _evaluate_point(
            positions=positions, coordinate_system=coordinate_system
        )
    elif dim == 1:
        return _evaluate_line(
            positions=positions,
            coordinate_system=coordinate_system,
            neighbour_mapping=neighbour_mapping,
            feature_size=feature_size
        )
    else:
        return None


def _evaluate_point(positions, coordinate_system):
    return NodalPoint(
        position=coordinate_system.
        average([np.array(pos) for pos in positions])
    )


class NodalPoint(SimpleNamespace):
    def __init__(self, position):
        self.position = position


def _evaluate_line(
    positions, coordinate_system, neighbour_mapping, feature_size
):
    # positions = list(positions)

    pos1 = positions.pop()
    for pos_candidate in positions:
        if (
            coordinate_system.distance(
                np.array(pos1), np.array(pos_candidate)
            ) > 2 * feature_size
        ):
            pos2 = pos_candidate
            positions.remove(pos2)
            break
    else:
        raise ValueError('No suitable second position found.')
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
    path_full = path1 + list(reversed(path2))
    return NodalLine(path=path_full)


def _get_shortest_path(
    *,
    positions,
    index_mapping,
    neighbour_mapping,
    weight_func=lambda dist: 1 / dist
):

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


class NodalLine(SimpleNamespace):
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return 'NodalLine(path=<{} values>)'.format(len(self.path))