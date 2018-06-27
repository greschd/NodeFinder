from types import SimpleNamespace

from fsc.export import export

from ..search._controller import ControllerState

from ._cluster import create_clusters
from ._dimension import calculate_dimension
from ._evaluate import evaluate_cluster


@export
def run(result, feature_size=2e-3):
    if isinstance(result, ControllerState):
        result = result.result
    positions = [node.pos for node in result.nodes]
    coordinate_system = result.coordinate_system
    return run_from_positions(
        positions=positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size
    )


@export
def run_from_positions(positions, *, coordinate_system, feature_size=2e-3):
    clusters, neighbour_mapping = create_clusters(
        positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size
    )
    results = []
    for cl in clusters:
        # TODO: use 'coordinate_system' to determine the dimension.
        dim = calculate_dimension(
            positions=cl,
            neighbour_mapping=neighbour_mapping,
            feature_size=feature_size
        )
        res = IdentificationResult(
            positions=cl,
            dimension=dim,
            result=evaluate_cluster(
                positions=cl,
                dim=dim,
                coordinate_system=coordinate_system,
                neighbour_mapping=neighbour_mapping
            )
        )
        results.append(res)
    return results


class IdentificationResult(SimpleNamespace):
    def __init__(self, positions, dimension, result=None):
        self.positions = positions
        self.dimension = dimension
        self.result = result

    def __repr__(self):
        return 'IdentificationResult(dimension={}, result={}, positions=<{} values>)'.format(
            self.dimension, self.result, len(self.positions)
        )
