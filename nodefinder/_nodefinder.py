"""
Implements the node finding algorithm.
"""

import asyncio
import itertools

import numpy as np
import fsc.hdf5_io
from fsc.export import export

from ._logging import _LOGGER
from ._result import NodeFinderResult, StartingPoint, NodalPoint
from ._nelder_mead import root_nelder_mead
from ._batch_submit import BatchSubmitter


@export
class NodeFinder:
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
        self,
        gap_fct,
        *,
        fct_listable=True,
        gap_threshold=1e-6,
        feature_size=1e-3,
        refinement_box_size=5e-3,
        refinement_mesh_size=(2, 2, 2),
        initial_result=None,
        save_file=None,
        load=False,
        load_quiet=True,
        initial_box_position=((0, 1), ) * 3,
        initial_mesh_size=(10, 10, 10),
        force_initial_mesh=False,
        num_minimize_parallel=50,
        **nelder_mead_kwargs
    ):
        if fct_listable:
            listable_gap_fct = gap_fct
        else:
            listable_gap_fct = lambda input_list: [gap_fct(x) for x in input_list]

        self._batch_submitter = BatchSubmitter(listable_gap_fct)
        self._func = self._batch_submitter.submit

        self._refinement_dist = refinement_box_size / 2
        self._refinement_mesh_size = refinement_mesh_size

        self._save_file = save_file
        self._create_result(
            feature_size=feature_size,
            gap_threshold=gap_threshold,
            initial_result=initial_result,
            load=load,
            load_quiet=load_quiet,
            initial_mesh_size=initial_mesh_size,
            initial_box_position=initial_box_position,
            force_initial_mesh=force_initial_mesh,
        )
        self._num_minimize_parallel = num_minimize_parallel
        self._nelder_mead_kwargs = nelder_mead_kwargs
        self._needs_saving = False

    def _create_result(  # pylint: disable=too-many-arguments
        self, feature_size, gap_threshold, initial_result, load, load_quiet,
        initial_mesh_size, initial_box_position, force_initial_mesh
    ):
        if load and initial_result:
            raise ValueError("Cannot set both 'load=True' and 'init_result'.")
        if load:
            try:
                initial_result = fsc.hdf5_io.load(self._save_file)
            except IOError as exc:
                if not load_quiet:
                    raise exc

        initial_mesh_points = self._get_box_starting_points(
            mesh_size=initial_mesh_size,
            box_position=initial_box_position,
        )
        if initial_result:
            self._result = NodeFinderResult(
                gap_threshold=gap_threshold,
                feature_size=feature_size,
                nodal_points=initial_result.nodal_points,
                starting_points=initial_result.starting_points
            )
            if force_initial_mesh:
                self._result.add_starting_points(initial_mesh_points)
        else:
            self._result = NodeFinderResult(
                gap_threshold=gap_threshold,
                feature_size=feature_size,
                starting_points=initial_mesh_points
            )

    def run(self):
        loop = asyncio.get_event_loop()
        with self._batch_submitter:
            loop.run_until_complete(self._run(loop))
        return self._result

    async def _run(self, loop):
        save_task = loop.create_task(self._save_loop())
        all_minimize_tasks = []
        while not self._result.finished:
            while self._result.num_running < self._num_minimize_parallel:
                if self._result.has_queued_points:
                    starting_point = self._result.pop_queued_starting_point()
                    _LOGGER.info(
                        'Submitting minimization with starting point k={}.'.
                        format(starting_point.k)
                    )
                    all_minimize_tasks.append(
                        loop.create_task(
                            self._run_starting_point(starting_point)
                        )
                    )
                else:
                    break
            # check for exceptions
            for task in all_minimize_tasks:
                try:
                    exc = task.exception()
                except asyncio.InvalidStateError:
                    continue
                if exc is not None:
                    raise exc
            await asyncio.sleep(0)
        await asyncio.gather(*all_minimize_tasks)
        save_task.cancel()

    @staticmethod
    def _get_box_starting_points(*, box_position, mesh_size):
        periodic = np.allclose(box_position, [[0, 1]] * len(box_position))
        return [
            StartingPoint(k=k)
            for k in itertools.product(
                *[
                    np.linspace(min_val, max_val, N, endpoint=not periodic)
                    for (min_val, max_val), N in zip(box_position, mesh_size)
                ]
            )
        ]

    async def _run_starting_point(self, starting_point):
        res = await self._minimize(starting_point)
        k = res.x
        is_new_node = self._result.add_result(
            starting_point=starting_point,
            nodal_point=NodalPoint(k=k, gap=res.fun)
        )
        if is_new_node:
            self._result.add_starting_points(
                self._get_box_starting_points(
                    box_position=[(
                        ki - self._refinement_dist, ki + self._refinement_dist
                    ) for ki in k],
                    mesh_size=self._refinement_mesh_size
                )
            )
        self._needs_saving = True

    async def _minimize(self, starting_point):
        """
        Minimize the gap function from the given starting point.
        """
        # TODO:
        # * Change the minimization to contain the dynamic cutoff criterion
        # * Make cutoff values configurable
        # * Allow setting the other starting vertices of the Nelder-Mead algorithm
        # res = so.minimize(self.gap_fct, x0=starting_point, method='Nelder-Mead', tol=1e-8, options=dict(maxfev=20))
        # if res.fun < 0.1:
        #     res = so.minimize(self.gap_fct, x0=res.x, method='Nelder-Mead', tol=1e-8, options=dict(maxfev=100))
        #     if res.fun < 1e-2:
        return await root_nelder_mead(
            self._func, x0=starting_point.k, **self._nelder_mead_kwargs
        )

    async def _save_loop(self):
        try:
            if not self._save_file:
                return
            while True:
                await asyncio.sleep(1.)
                self._do_save()
        except asyncio.CancelledError:
            self._do_save()

    def _do_save(self):
        if self._needs_saving:
            fsc.hdf5_io.save(self._result, self._save_file)
