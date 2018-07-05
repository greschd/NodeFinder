def _setup_plot(limits, axis=None):
    """Sets up the figure axes and handles options common to all plots."""
    # import is here s.t. the import of the package does not fail
    # if matplotlib is not present
    import matplotlib.pyplot as plt

    dim = len(limits)
    if dim == 3:
        is_3d = True
        from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
    else:
        assert dim == 2
    # create axis if it does not exist
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d' if is_3d else None)
    else:
        fig = None

    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    if is_3d:
        axis.set_zlim(*limits[2])

    return fig, axis, is_3d
