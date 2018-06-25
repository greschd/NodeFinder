import os
import asyncio
import tempfile
import itertools

import numpy as np
from fsc.export import export
import fsc.hdf5_io
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5
from fsc.async_tools import PeriodicTask, wrap_to_coroutine

from ._queue import SimplexQueue
from ._result import ResultContainer
from ._coordinate_system import CoordinateSystem
from ._minimization import run_minimization


@export
@subscribe_hdf5('nodefinder.controller_state')
class ControllerState(HDF5Enabled):
    def __init__(self, *, result, queue):
        self.result = result
        self.queue = queue

    def to_hdf5(self, hdf5_handle):
        result_group = hdf5_handle.create_group('result')
        to_hdf5(self.result, result_group)
        queue_group = hdf5_handle.create_group('queue')
        to_hdf5(self.queue, queue_group)

    @classmethod
    def from_hdf5(cls, hdf5_handle):
        return cls(
            result=from_hdf5(hdf5_handle['result']),
            queue=from_hdf5(hdf5_handle['queue']),
        )


class Controller:
    """

    Arguments
    ---------
    gap_fct : Callable
        Function that returns the gap, given a k-point.
    gap_threshold : float
        Threshold when the gap is considered to be closed.
    feature_size : float
        Minimum distance between nodal features for them to be considered distinct.
    initial_box_position : tuple[tuple[float]]
        Initial box within which the minimization starting points are selected.
    initial_mesh_size : tuple[int]
        Initial mesh of starting points.
    """

    def __init__(
        self, *, gap_fct, limits, initial_state, save_file, load, load_quiet,
        initial_mesh_size, force_initial_mesh, gap_threshold, feature_size,
        fake_potential_class, nelder_mead_kwargs, num_minimize_parallel,
        refinement_box_size, refinement_mesh_size
    ):
        self.gap_fct = wrap_to_coroutine(gap_fct)

        self.coordinate_system = CoordinateSystem(limits=limits, periodic=True)
        self.dim = self.check_dimensions(
            limits, initial_mesh_size, refinement_mesh_size
        )
        self.save_file = save_file
        self.state = self.create_state(
            initial_state=initial_state,
            load=load,
            load_quiet=load_quiet,
            initial_mesh_size=initial_mesh_size,
            force_initial_mesh=force_initial_mesh,
            gap_threshold=gap_threshold,
            dist_cutoff=feature_size
        )
        self.feature_size = feature_size
        if fake_potential_class is not None:
            self.fake_potential = fake_potential_class(
                result=self.state.result,
                width=self.feature_size,
                height=gap_threshold
            )
        else:
            self.fake_potential = None
        self.refinement_stencil = self.create_refinement_stencil(
            refinement_box_size=refinement_box_size,
            refinement_mesh_size=refinement_mesh_size
        )
        self.num_minimize_parallel = num_minimize_parallel
        self.nelder_mead_kwargs = dict(nelder_mead_kwargs)

        self.task_futures = set()

    @staticmethod
    def check_dimensions(limits, mesh_size, refinement_mesh_size):
        dim_limits = len(limits)
        dim_mesh_size = len(mesh_size)
        dim_refinement_mesh_size = len(refinement_mesh_size)
        if not dim_limits == dim_mesh_size == dim_refinement_mesh_size:
            raise ValueError(
                'Inconsistent dimensions given: limits: {}, mesh_size: {}, refinement_mesh_size: {}'.
                format(dim_limits, dim_mesh_size, dim_refinement_mesh_size)
            )
        return dim_limits

    def create_state(
        self,
        *,
        initial_state,
        load,
        load_quiet,
        initial_mesh_size,
        force_initial_mesh,
        gap_threshold,
        dist_cutoff,
    ):
        if load:
            if initial_state is not None:
                raise ValueError(
                    "Cannot set the initial state explicitly and setting 'load=True' simulatneously."
                )
            try:
                initial_state = fsc.hdf5_io.load(self.save_file)
            except IOError as exc:
                if not load_quiet:
                    raise exc
        if initial_state is not None:
            result = ResultContainer(
                coordinate_system=self.coordinate_system,
                minimization_results=initial_state.result.minimization_results,
                gap_threshold=gap_threshold,
                dist_cutoff=dist_cutoff
            )
            queue = SimplexQueue(simplices=initial_state.queue.simplices)
            if force_initial_mesh:
                queue.add_simplices(
                    self.get_initial_simplices(
                        initial_mesh_size=initial_mesh_size
                    )
                )
        else:
            result = ResultContainer(
                coordinate_system=self.coordinate_system,
                gap_threshold=gap_threshold,
                dist_cutoff=dist_cutoff,
            )
            queue = SimplexQueue(self.get_initial_simplices(initial_mesh_size))
        return ControllerState(result=result, queue=queue)

    def get_initial_simplices(self, initial_mesh_size):
        return self.generate_simplices(
            limits=self.coordinate_system.limits, mesh_size=initial_mesh_size
        )

    def generate_simplices(self, limits, mesh_size):
        vertices = list(
            itertools.product(
                *[
                    np.linspace(lower, upper, m)
                    for (lower, upper), m in zip(limits, mesh_size)
                ]
            )
        )
        size = np.array([upper - lower for lower, upper in limits])
        simplex_distances = size / (2 * np.array(mesh_size))
        simplex_stencil = np.zeros(shape=(self.dim + 1, self.dim))
        for i, dist in enumerate(simplex_distances):
            simplex_stencil[i + 1][i] = dist
        return [v + simplex_stencil for v in vertices]

    def create_refinement_stencil(
        self, refinement_box_size, refinement_mesh_size
    ):
        half_size = refinement_box_size / 2
        return np.array(
            self.generate_simplices(
                limits=[(-half_size, half_size)] * self.dim,
                mesh_size=refinement_mesh_size
            )
        )

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.create_tasks())

    async def create_tasks(self):
        async with PeriodicTask(self.save, delay=5.):
            while not self.state.queue.finished:
                while (
                    self.state.queue.has_queued_points and
                    self.state.queue.num_running < self.num_minimize_parallel
                ):
                    simplex = self.state.queue.pop_queued()
                    self.schedule_minimization(simplex)
                await asyncio.sleep(0.)
                # retrieve exceptions
                for fut in list(self.task_futures):
                    if fut.done():
                        await fut
                        self.task_futures.remove(fut)
        await asyncio.gather(*self.task_futures)

    def schedule_minimization(self, simplex):
        self.task_futures.add(asyncio.ensure_future(self.run_simplex(simplex)))

    async def run_simplex(self, simplex):
        result = await run_minimization(
            self.gap_fct,
            initial_simplex=simplex,
            fake_potential=self.fake_potential,
            nelder_mead_kwargs=self.nelder_mead_kwargs
        )
        self.process_result(result)
        self.state.queue.set_finished(simplex)

    def process_result(self, result):
        is_node = self.state.result.add_result(result)
        if is_node:
            pos = result.pos
            if all(
                dist >= self.feature_size
                for dist in
                self.state.result.get_neighbour_distance_iterator(pos)
            ):
                self.state.queue.add_simplices(pos + self.refinement_stencil)

    def save(self):
        if self.save_file:
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(self.save_file), delete=False
            ) as tmpf:
                try:
                    fsc.hdf5_io.save(self.state, tmpf.name)
                    os.rename(tmpf.name, self.save_file)
                except Exception as exc:
                    os.remove(tmpf.name)
                    raise exc
