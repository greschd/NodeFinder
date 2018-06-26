from types import MappingProxyType

from fsc.export import export

from ._controller import Controller


@export
def run(
    gap_fct,
    *,
    limits=((0, 1), ) * 3,
    initial_state=None,
    save_file=None,
    load=False,
    load_quiet=True,
    initial_mesh_size=(10, 10, 10),
    force_initial_mesh=False,
    refinement_box_size=None,
    refinement_mesh_size=(3, 3, 3),
    gap_threshold=1e-6,
    feature_size=2e-3,
    use_fake_potential=False,
    nelder_mead_kwargs=MappingProxyType({}),
    num_minimize_parallel=50,
):
    controller = Controller(
        gap_fct=gap_fct,
        limits=limits,
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
    controller.run()
    return controller.state.result
