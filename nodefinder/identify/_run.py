"""
Defines the function which runs the identify step.
"""

from types import SimpleNamespace

from fsc.export import export
from fsc.hdf5_io import subscribe_hdf5, SimpleHDF5Mapping

from ..search._controller import ControllerState, _DIST_CUTOFF_FACTOR

from ._cluster import create_clusters
from ._dimension import calculate_dimension
from ._evaluate import evaluate_cluster


@export
def run(result, feature_size=None):
    """Identify the nodal clusters from a :func:`.search.run` result.

    Arguments
    ---------
    result : SearchResultContainer
        Result of the search step.
    feature_size : float, optional
        TODO. Uses the ``feature_size`` used in the search step by default.
    """
    if feature_size is None:
        feature_size = result.dist_cutoff * _DIST_CUTOFF_FACTOR
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
def run_from_positions(positions, *, coordinate_system, feature_size):
    """
    """
    clusters, neighbour_mapping = create_clusters(
        positions,
        coordinate_system=coordinate_system,
        feature_size=feature_size
    )
    results = []
    for cluster in clusters:
        # TODO: use 'coordinate_system' to determine the dimension.
        dim = calculate_dimension(
            positions=cluster,
            neighbour_mapping=neighbour_mapping,
            coordinate_system=coordinate_system,
            feature_size=feature_size
        )
        res = IdentificationResult(
            positions=cluster,
            dimension=dim,
            shape=evaluate_cluster(
                positions=cluster,
                dim=dim,
                coordinate_system=coordinate_system,
                neighbour_mapping=neighbour_mapping,
                feature_size=feature_size,
            )
        )
        results.append(res)
    return IdentificationResultContainer(
        coordinate_system=coordinate_system,
        feature_size=feature_size,
        results=results
    )


@export
@subscribe_hdf5('nodefinder.identification_result_container')
class IdentificationResultContainer(SimpleNamespace, SimpleHDF5Mapping):
    HDF5_ATTRIBUTES = ['coordinate_system', 'results', 'feature_size']

    def __init__(self, *, coordinate_system, feature_size, results=()):
        self.coordinate_system = coordinate_system
        self.results = results
        self.feature_size = feature_size

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx):
        return self.results[idx]

    def __len__(self):
        return len(self.results)


@export
@subscribe_hdf5('nodefinder.identification_result')
class IdentificationResult(SimpleNamespace, SimpleHDF5Mapping):
    HDF5_ATTRIBUTES = ['positions', 'shape', 'dimension']

    def __init__(self, positions, dimension, shape=None):
        self.positions = positions
        self.dimension = dimension
        self.shape = shape

    def __repr__(self):
        return 'IdentificationResult(dimension={}, shape={}, positions=<{} values>)'.format(
            self.dimension, self.shape, len(self.positions)
        )
