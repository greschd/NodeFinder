"""
Defines the function which runs the search step.
"""

import queue
import asyncio
import threading
from functools import partial
from types import MappingProxyType

from fsc.export import export

from ._controller import Controller
from ._logging import SEARCH_LOGGER


@export
async def run_async(
    gap_fct,
    *,
    limits=((0, 1), ) * 3,
    periodic=True,
    initial_state=None,
    save_file=None,
    load=False,
    load_quiet=True,
    initial_mesh_size=10,
    force_initial_mesh=False,
    refinement_box_size=None,
    refinement_mesh_size=3,
    gap_threshold=1e-6,
    feature_size=2e-3,
    use_fake_potential=False,
    nelder_mead_kwargs=MappingProxyType({}),
    num_minimize_parallel=50
):
    """Run the nodal point search.

    Arguments
    ---------
    gap_fct : collections.abc.Callable
        Function or coroutine describing the potential of which nodes should be
        found.
    limits : tuple(tuple(float))
        The limits of the box where nodes are searched, given as tuple for each
        dimension.
    periodic : bool
        Indicates whether periodic boundary conditions are used for the
        coordinate system.
    save_file : str
        Path to the file where the intermediate results are stored.
    load : bool
        Enable or disable loading the initial state from ``save_file``.
    load_quiet : bool
        When set to ``True``, ignore errors when loading the initial state.
    initial_mesh_size : int or tuple(int)
        Size of the initial mesh of starting points. Can be given either as a
        single integer or as a list of integers corresponding to the different
        dimensions.
    force_initial_mesh : bool
        If ``True``, add the initial mesh also when restarting the calculation
        from an intermediate result.
    refinement_box_size : float
        Size of the box where refinement starting points are placed after
        finding a new node.
    refinement_mesh_size : int or tuple(int)
        Mesh size for the refinement step.
    gap_threshold : float
        Threshold for the function value for which a given point is considered
        to be a node.
    feature_size : float
        Threshold for the distance between two nodes where they are considered
        distinct.
    use_fake_potential : bool
        If ``True``, the minimization for a given simplex is performed in two
        steps, first adding a fake potential to repel the minimization from
        existing nodes.
    nelder_mead_kwargs : collections.abc.Mapping
        Keyword arguments passed to the Nelder-Mead algorithm.
    num_minimize_parallel : int
        Maximum number of minimization calculations which are launched in
        parallel.

    Returns
    -------
    SearchResultContainer:
        The result of the search algorithm.
    """
    SEARCH_LOGGER.debug('Initializing search controller.')
    controller = Controller(
        gap_fct=gap_fct,
        limits=limits,
        periodic=periodic,
        initial_state=initial_state,
        save_file=save_file,
        load=load,
        load_quiet=load_quiet,
        initial_mesh_size=initial_mesh_size,
        force_initial_mesh=force_initial_mesh,
        gap_threshold=gap_threshold,
        feature_size=feature_size,
        use_fake_potential=use_fake_potential,
        nelder_mead_kwargs=nelder_mead_kwargs,
        num_minimize_parallel=num_minimize_parallel,
        refinement_box_size=refinement_box_size,
        refinement_mesh_size=refinement_mesh_size
    )
    SEARCH_LOGGER.debug('Running search controller.')
    await controller.run()
    SEARCH_LOGGER.debug('Search controller finished.')
    return controller.state.result


@export
def run(*args, **kwargs):
    """Wrapper around :func:`.run_async` that runs the node search synchronously.

    Arguments
    ---------
    args : tuple
        Positional arguments passed to :func:`.run_async`.
    kwargs : collections.abc.Mapping
        Keyword arguments passed to :func:`.run_async`.
    """
    try:
        loop = asyncio.get_event_loop()
        close_loop = False
    except RuntimeError:
        SEARCH_LOGGER.debug('Creating a new event loop.')
        loop = asyncio.new_event_loop()
        close_loop = True

    try:
        if loop.is_running():
            SEARCH_LOGGER.debug('Running in a separate thread.')
            res_queue = queue.Queue()
            exc_queue = queue.Queue()
            thread = threading.Thread(
                target=partial(
                    _run_in_thread,
                    *args,
                    res_queue=res_queue,
                    exc_queue=exc_queue,
                    **kwargs
                )
            )
            SEARCH_LOGGER.debug('Starting thread.')
            thread.start()
            SEARCH_LOGGER.debug('Joining thread.')
            thread.join()
            if not exc_queue.empty():
                raise exc_queue.get()
            res = res_queue.get()
        else:
            SEARCH_LOGGER.debug('Running in the current thread.')
            res = loop.run_until_complete(run_async(*args, **kwargs))
    finally:
        if close_loop:
            loop.close()
    return res


def _run_in_thread(*args, res_queue, exc_queue, **kwargs):
    """
    Helper function that runs the search function, to be used as a thread target.
    This assumes that no (running or other) loop exists.
    """
    try:
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(run_async(*args, **kwargs))
        loop.close()
        res_queue.put(res)
    except Exception as exc:
        exc_queue.put(exc)
        raise
