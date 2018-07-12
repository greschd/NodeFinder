"""
Defines the Controller, which implements the evaluation of the search step.
"""

import os
import numbers
import asyncio
import tempfile
import itertools
from collections import ChainMap

import numpy as np

from fsc.async_tools import PeriodicTask, wrap_to_coroutine

from .. import io
from ..coordinate_system import CoordinateSystem
from .result import SearchResultContainer, ControllerState
from ._queue import SimplexQueue
from ._minimization import run_minimization
from ._fake_potential import FakePotential

_DIST_CUTOFF_FACTOR = 3


class Controller:
    """
    Implementation class for the :func:`.search.run` function.

    Arguments are the same as defined in :func:`.search.run`.
    """

    def __init__(
        self,
        *,
        gap_fct,
        limits,
        periodic,
        initial_state,
        save_file,
        load,
        load_quiet,
        initial_mesh_size,
        force_initial_mesh,
        gap_threshold,
        feature_size,
        nelder_mead_kwargs,
        num_minimize_parallel,
        refinement_box_size,
        refinement_mesh_size,
        use_fake_potential=True
    ):
        self.gap_fct = wrap_to_coroutine(gap_fct)

        self.coordinate_system = CoordinateSystem(
            limits=limits, periodic=periodic
        )
        self.dim, initial_mesh_size, refinement_mesh_size = self.check_dimensions(
            limits, initial_mesh_size, refinement_mesh_size
        )
        self.save_file = save_file
        self.dist_cutoff = feature_size / _DIST_CUTOFF_FACTOR
        self.state = self.create_state(
            initial_state=initial_state,
            load=load,
            load_quiet=load_quiet,
            initial_mesh_size=initial_mesh_size,
            force_initial_mesh=force_initial_mesh,
            gap_threshold=gap_threshold,
            dist_cutoff=self.dist_cutoff
        )
        if use_fake_potential:
            self.fake_potential = FakePotential(
                result=self.state.result,
                width=self.dist_cutoff,
            )
        else:
            self.fake_potential = None
        self.refinement_stencil = self.create_refinement_stencil(
            refinement_box_size=refinement_box_size or 5 * self.dist_cutoff,
            refinement_mesh_size=refinement_mesh_size
        )
        self.num_minimize_parallel = num_minimize_parallel
        self.nelder_mead_kwargs = ChainMap(
            nelder_mead_kwargs, {
                'ftol': 0.05 * gap_threshold,
                'xtol': 0.03 * self.dist_cutoff
            }
        )

        self.task_futures = set()

    @staticmethod
    def check_dimensions(limits, mesh_size, refinement_mesh_size):
        """
        Check that the dimensions of the given inputs match.
        """
        if isinstance(mesh_size, numbers.Integral):
            mesh_size = tuple(mesh_size for _ in range(len(limits)))
        if isinstance(refinement_mesh_size, numbers.Integral):
            refinement_mesh_size = tuple(
                refinement_mesh_size for _ in range(len(limits))
            )
        dim_limits = len(limits)
        dim_mesh_size = len(mesh_size)
        dim_refinement_mesh_size = len(refinement_mesh_size)
        if not dim_limits == dim_mesh_size == dim_refinement_mesh_size:
            raise ValueError(
                'Inconsistent dimensions given: limits: {}, mesh_size: {}, refinement_mesh_size: {}'.
                format(dim_limits, dim_mesh_size, dim_refinement_mesh_size)
            )
        return dim_limits, mesh_size, refinement_mesh_size

    def create_state(
        self, *, initial_state, load, load_quiet, initial_mesh_size,
        force_initial_mesh, gap_threshold, dist_cutoff
    ):
        """
        Load or create the initial state of the calculation.
        """
        if load:
            if initial_state is not None:
                raise ValueError(
                    "Cannot set the initial state explicitly and setting 'load=True' simultaneously."
                )
            try:
                initial_state = io.load(self.save_file)
            except IOError as exc:
                if not load_quiet:
                    raise exc
        if initial_state is not None:
            result = SearchResultContainer(
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
            result = SearchResultContainer(
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
        """
        Generate the starting simplices for given limits and mesh size.
        """
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
        """
        Create a stencil for the simplices used in the refinement step.
        """
        if np.product(refinement_mesh_size) == 0:
            return None
        half_size = refinement_box_size / 2
        return np.array(
            self.generate_simplices(
                limits=[(-half_size, half_size)] * self.dim,
                mesh_size=refinement_mesh_size
            )
        )

    async def run(self):
        await self.create_tasks()

    async def create_tasks(self):
        """
        Create minimization tasks until the calculation is finished.
        """
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
        """
        Run the minimization for a given starting simplex.
        """
        result = await run_minimization(
            self.gap_fct,
            initial_simplex=simplex,
            fake_potential=self.fake_potential,
            nelder_mead_kwargs=self.nelder_mead_kwargs,
        )
        self.process_result(result)
        self.state.queue.set_finished(simplex)

    def process_result(self, result):
        is_node = self.state.result.add_result(result)
        if is_node and self.refinement_stencil is not None:
            pos = result.pos
            if all(
                dist >= self.dist_cutoff for dist in
                self.state.result.get_neighbour_distance_iterator(pos)
            ):
                self.state.queue.add_simplices(pos + self.refinement_stencil)

    def save(self):
        if self.save_file:
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(self.save_file), delete=False
            ) as tmpf:
                try:
                    io.save(self.state, tmpf.name)
                    os.rename(tmpf.name, self.save_file)
                except Exception as exc:
                    os.remove(tmpf.name)
                    raise exc
