from fsc.export import export

from ..search._controller import ControllerState

from ._cluster import create_clusters


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
    print(len(clusters))
    # print(clusters, neighbour_mapping)
