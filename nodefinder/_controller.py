import asyncio
import itertools
from collections import ChainMap

import numpy as np
from fsc.export import export
import fsc.hdf5_io
from fsc.hdf5_io import HDF5Enabled, subscribe_hdf5, to_hdf5, from_hdf5
from fsc.async_tools import PeriodicTask, wrap_to_coroutine

from ._queue import SimplexQueue
from ._result import Result
from ._coordinate_system import CoordinateSystem
from ._minimization import run_minimization


@export
@subscribe_hdf5('nodefinder.controller_state')
class ControllerState(HDF5Enabled):
    def __init__(self, *, result, queue):
        self.result = result
        self.queue = queue

    def __getattr__(self, key):
        try:
            return getattr(self.result, key)
        except AttributeError:
            return getattr(self.queue, key)

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
        fake_potential, nelder_mead_kwargs, num_minimize_parallel
    ):
        self.gap_fct = wrap_to_coroutine(gap_fct)
        self.fake_potential = fake_potential
        self.coordinate_system = CoordinateSystem(limits=limits, periodic=True)
        self.save_file = save_file
        self.state = self.create_state(
            initial_state=initial_state,
            load=load,
            load_quiet=load_quiet,
            initial_mesh_size=initial_mesh_size,
            force_initial_mesh=force_initial_mesh,
            gap_threshold=gap_threshold,
            dist_cutoff=getattr(self.fake_potential, 'dist_cutoff', 0.)
        )
        self.num_minimize_parallel = num_minimize_parallel
        self.nelder_mead_kwargs = ChainMap(
            nelder_mead_kwargs,
            dict(ftol=0.1 * gap_threshold, xtol=0.1 * feature_size)
        )

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
            result = Result(
                coordinate_system=self.coordinate_system,
                minimization_results=initial_state.result.minimization_results,
                gap_threshold=gap_threshold,
                dist_cutoff=dist_cutoff
            )
            queue = SimplexQueue(initial_state.queue)
            if force_initial_mesh:
                queue.add_simplices(
                    self.get_initial_simplices(
                        initial_mesh_size=initial_mesh_size
                    )
                )
        else:
            result = Result(
                coordinate_system=self.coordinate_system,
                gap_threshold=gap_threshold,
                dist_cutoff=dist_cutoff,
            )
            queue = SimplexQueue(self.get_initial_simplices(initial_mesh_size))
        return ControllerState(result=result, queue=queue)

    def get_initial_simplices(self, initial_mesh_size):
        vertices_frac = list(
            itertools.
            product(*[np.linspace(0, 1, m) for m in initial_mesh_size])
        )
        dim = len(initial_mesh_size)
        simplex_distances = 1 / (2 * np.array(initial_mesh_size))
        simplex_stencil = np.zeros(shape=(dim + 1, dim))
        for i, dist in enumerate(simplex_distances):
            simplex_stencil[i + 1][i] = dist
        return [
            self.coordinate_system.get_pos(v + simplex_stencil)
            for v in vertices_frac
        ]

    def run(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.create_tasks())

    async def create_tasks(self):
        futures = set()
        async with PeriodicTask(self.save, delay=5.):
            while not self.state.queue.finished:
                while (
                    self.state.queue.has_queued_points and
                    self.state.queue.num_running < self.num_minimize_parallel
                ):
                    simplex = self.state.queue.pop_queued()
                    futures.add(
                        asyncio.ensure_future(self.run_simplex(simplex))
                    )
                await asyncio.sleep(0.)
                # print('not finished', self.state.queue.num_running, len(self.state.queue._queued_simplices))
                # retrieve exceptions
                for fut in futures:
                    if fut.done():
                        await fut
        await asyncio.gather(*futures)

    async def run_simplex(self, simplex):
        # print('running')
        result = await run_minimization(
            self.gap_fct,
            initial_simplex=simplex,
            fake_potential=self.fake_potential,
            nelder_mead_kwargs=self.nelder_mead_kwargs
        )
        # print('done')
        self.process_result(result)
        self.state.queue.set_finished(simplex)

    def process_result(self, result):
        self.state.result.add_result(result)
        # TODO: Implement refinement

    def save(self):
        if self.save_file:
            fsc.hdf5_io.save(self.state, self.save_file)
