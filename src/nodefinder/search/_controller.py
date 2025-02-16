# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""
Defines the Controller, which implements the evaluation of the search step.
"""

import os
import numbers
import asyncio
import tempfile
from collections import ChainMap

import numpy as np
from fsc.export import export
from fsc.async_tools import PeriodicTask, wrap_to_coroutine

from .. import io
from ..coordinate_system import CoordinateSystem
from .result import SearchResultContainer, ControllerState
from ._queue import SimplexQueue, PositionQueue
from ._minimization import run_minimization
from ._fake_potential import FakePotential
from ._logging import SEARCH_LOGGER
from ._mesh_helper import _generate_mesh_simplices
from .refinement_stencil import get_auto_stencil

_DIST_CUTOFF_FACTOR = 3


@export
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
        save_delay=5.0,
        load,
        load_quiet,
        initial_mesh_size,
        force_initial_mesh,
        gap_threshold,
        feature_size,
        nelder_mead_kwargs,
        num_minimize_parallel,
        refinement_stencil,
        use_fake_potential=True,
        recheck_pos_dist=True,
        recheck_count_cutoff=0,
        simplex_check_cutoff=0,
    ):
        self.gap_fct = wrap_to_coroutine(gap_fct)

        self.coordinate_system = CoordinateSystem(limits=limits, periodic=periodic)
        self.dim, initial_mesh_size = self.check_dimensions(limits, initial_mesh_size)
        self.save_file = save_file
        self.save_delay = save_delay

        self.dist_cutoff = feature_size / _DIST_CUTOFF_FACTOR
        self.state = self.create_state(
            initial_state=initial_state,
            load=load,
            load_quiet=load_quiet,
            initial_mesh_size=initial_mesh_size,
            force_initial_mesh=force_initial_mesh,
            gap_threshold=gap_threshold,
            dist_cutoff=self.dist_cutoff,
        )
        if use_fake_potential:
            self.fake_potential = FakePotential(
                result=self.state.result,
                width=self.dist_cutoff,
            )
        else:
            self.fake_potential = None

        if isinstance(refinement_stencil, str) and refinement_stencil == "auto":
            refinement_stencil = get_auto_stencil(dim=self.dim)
        if refinement_stencil is not None:
            self.refinement_stencil = refinement_stencil * self.dist_cutoff
        else:
            self.refinement_stencil = None
        self.num_minimize_parallel = num_minimize_parallel
        self.nelder_mead_kwargs = ChainMap(
            nelder_mead_kwargs,
            {"ftol": 0.05 * gap_threshold, "xtol": 0.03 * self.dist_cutoff},
        )

        self.task_futures = set()
        self.recheck_pos_dist = recheck_pos_dist
        self.recheck_count_cutoff = recheck_count_cutoff
        self.simplex_check_cutoff = simplex_check_cutoff

    @staticmethod
    def check_dimensions(limits, mesh_size):
        """
        Check that the dimensions of the given inputs match.
        """
        if isinstance(mesh_size, numbers.Integral):
            mesh_size = tuple(mesh_size for _ in range(len(limits)))
        dim_limits = len(limits)
        dim_mesh_size = len(mesh_size)
        if not dim_limits == dim_mesh_size:
            raise ValueError(
                "Inconsistent dimensions given: limits: {}, mesh_size: {}".format(
                    dim_limits, dim_mesh_size
                )
            )
        return dim_limits, mesh_size

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
                dist_cutoff=dist_cutoff,
                refined_results=initial_state.result.refined_results,
            )
            simplex_queue = SimplexQueue(objects=initial_state.simplex_queue.objects)
            position_queue = PositionQueue(objects=initial_state.position_queue.objects)
            if force_initial_mesh:
                simplex_queue.add_objects(
                    self.get_initial_simplices(initial_mesh_size=initial_mesh_size)
                )
        else:
            result = SearchResultContainer(
                coordinate_system=self.coordinate_system,
                gap_threshold=gap_threshold,
                dist_cutoff=dist_cutoff,
            )
            simplex_queue = SimplexQueue(self.get_initial_simplices(initial_mesh_size))
            position_queue = PositionQueue()
        return ControllerState(
            result=result, simplex_queue=simplex_queue, position_queue=position_queue
        )

    def get_initial_simplices(self, initial_mesh_size):
        return _generate_mesh_simplices(
            limits=self.coordinate_system.limits,
            mesh_size=initial_mesh_size,
            periodic=self.coordinate_system.periodic,
        )

    async def run(self):
        await self.create_tasks()

    async def create_tasks(self):
        """
        Create minimization tasks until the calculation is finished.
        """
        async with PeriodicTask(self.save, delay=self.save_delay):
            while (
                not self.state.simplex_queue.finished
            ) or self.state.position_queue.has_queued:
                # if (not self.state.simplex_queue.has_queued) and (self.state.position_queue.has_queued):
                while self.state.simplex_queue.num_running < self.num_minimize_parallel:
                    while not self.state.simplex_queue.has_queued:
                        if self.state.position_queue.has_queued:
                            pos = self.state.position_queue.pop_queued()
                            if (
                                not self.recheck_pos_dist
                            ) or self._check_pos_refinement(
                                pos, count_cutoff=self.recheck_count_cutoff
                            ):
                                self.state.simplex_queue.add_objects(
                                    pos + self.refinement_stencil
                                )
                                self.state.result.set_refined(np.array(pos))
                            else:
                                SEARCH_LOGGER.debug(
                                    "Discarding refinement of position {}".format(pos)
                                )
                        else:
                            break
                    if self.state.simplex_queue.has_queued:
                        simplex = self.state.simplex_queue.pop_queued()
                        if self._check_simplex(simplex):
                            self.schedule_minimization(simplex)
                        else:
                            self.state.simplex_queue.set_finished(simplex)
                    else:
                        break

                await asyncio.sleep(0.0)

                # Retrieve all exceptions, to avoid 'exception never retrieved'
                # warning, but raise only the first one.
                done_futures = [fut for fut in self.task_futures if fut.done()]
                exceptions = [fut.exception() for fut in done_futures]
                exceptions = [exc for exc in exceptions if exc is not None]
                if exceptions:
                    raise exceptions[0]

                self.task_futures.difference_update(done_futures)
        await asyncio.gather(*self.task_futures)

    def schedule_minimization(self, simplex):
        SEARCH_LOGGER.debug("Scheduling minimization of simplex {}".format(simplex))
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
        self.state.simplex_queue.set_finished(simplex)

    def process_result(self, result):
        """
        Update the state with a given result, and add new simplices if needed.
        """
        is_node = self.state.result.add_result(result)
        if is_node and self.refinement_stencil is not None:
            pos = result.pos
            SEARCH_LOGGER.info("Found node at position {}".format(pos))
            if self._check_pos_refinement(pos):
                SEARCH_LOGGER.info("Scheduling refinement around node.")
                self.state.position_queue.add_objects([pos])

    def _check_pos_refinement(self, pos, count_cutoff=0):
        """
        Check whether a given position should be scheduled for refinement. An
        optional cutoff for the number of positions which are allowed to be
        within the cutoff distance can be given.
        """
        count = 0
        for dist in self.state.result.get_refined_neighbour_distance_iterator(pos):
            if dist < self.dist_cutoff:
                count += 1
            if count > count_cutoff:
                return False
        return True

    def _check_simplex(self, simplex):
        """
        Check if a simplex should be evaluated. Returns False if all vertices
        of the simplex are within dist_cutoff from an existing node.
        """
        count = 0
        for pos in simplex:
            for dist in self.state.result.get_neighbour_distance_iterator(pos):
                if dist < self.dist_cutoff:
                    count += 1
                    if count > self.simplex_check_cutoff:
                        return False
                    break
        return True

    def save(self):
        """
        Store the current ControllerState to the save file.
        """
        if self.save_file and self.state.needs_saving:
            with tempfile.NamedTemporaryFile(
                dir=os.path.dirname(self.save_file), delete=False
            ) as tmpf:
                try:
                    io.save(self.state, tmpf.name)
                    os.rename(tmpf.name, self.save_file)
                    self.state.needs_saving = False
                except Exception as exc:
                    os.remove(tmpf.name)
                    raise exc
