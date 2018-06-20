import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import


def plot_3d(result, ax=None):  # pylint: disable=inconsistent-return-statements
    if ax is None:
        return_fig_ax = True
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        return_fig_ax = False

    x_coords = []
    y_coords = []
    z_coords = []
    vals = []
    for node in result.nodes:
        pos = node.pos
        x, y, z = pos
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        vals.append(node.value)

    ax.scatter(x_coords, y_coords, z_coords, c=vals)

    xlim, ylim, zlim = result.coordinate_system.limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    if return_fig_ax:
        return fig, ax
