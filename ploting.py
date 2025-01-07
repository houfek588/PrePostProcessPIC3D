import matplotlib.pyplot as plt
import numpy as np

plot_immediately = False

class PlotDescription:
    def __init__(self, title="title", label_x="axis x", label_y="axis y", label_z="axis z"):
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.label_z = label_z

        self.ymax = 0
        self.ymin = 0

    def set_ylim(self, minimum, maximum):
        self.ymax = maximum
        self.ymin = minimum




def plot_data(dataX, dataY, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    """
    Plots nodal accelerations from acceleration data.

    Args:
        acceleration_data (dict): Acceleration results from OP2 file.
    """
    scale = 2
    fig = plt.figure(figsize=(16/scale, 9/scale))

    if len(dataY) < 5:
        for i in range(0 ,len(dataY)):
            plt.plot(dataX, dataY[i])
            plt.scatter(dataX, dataY[i], color='red', label="Data Points", zorder=3)
    else:
        plt.plot(dataX, dataY)
        plt.scatter(dataX, dataY, color='red', label="Data Points", zorder=3)

    plt.xlabel(descr.label_x)
    plt.ylabel(descr.label_y)
    plt.title(descr.title)
    plt.grid(True)

    # Show data points


    if descr.ymax == 0 and descr.ymin == 0:
        print("y limits dont used")
        pass
    else:
        print(f"y limits ({descr.ymin}, {descr.ymax})")
        plt.ylim(descr.ymin, descr.ymax)

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300) # Save as PNG file

    if plot_immediately:
        plt.show()


def plot3D_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    # Create a figure and 3D axes
    scale = 1
    fig = plt.figure(figsize=(16/scale, 9/scale))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(dataX, dataY)
    Z = np.array(dataZ)
    # ground = np.zeros_like(X)

    # Plot a surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # surf = ax.plot_surface(X, Y, ground, facecolors=plt.cm.viridis((Z - Z.min()) / (Z.max() - Z.min())),
    #                        rstride=1, cstride=1, shade=False)
    # surf = ax.plot_surface(x_axis, time_scale, data_x_t, cmap='viridis')

    # for i in range(0, len(dataY), 20):
    #     # y = y_base + z * 0.1  # Optionally modify the curve (e.g., add offset)
    #     ax.plot(dataX, ys=dataY[i], zs=dataZ, zdir='y', label=f'Curve at y={dataY[i]:.1f}')

    # Add color bar
    fig.colorbar(surf, shrink=0.5, aspect=10)
    # mappable = plt.cm.ScalarMappable(cmap='viridis')
    # mappable.set_array(Z)
    # fig.colorbar(mappable, shrink=0.5, aspect=10, label="Color Scale")

    # Disable perspective by setting a large viewing distance
    ax.dist = 10  # The larger the value, the less perspective distortion

    # Add labels
    ax.set_title(descr.title)
    ax.set_xlabel(descr.label_x)
    ax.set_ylabel(descr.label_y)
    ax.set_zlabel(descr.label_z)
    # ax.view_init(elev=90, azim=0)

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300)  # Save as PNG file

    if plot_immediately:
        plt.show()


def plot3Dwire_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    # Create a figure and 3D axes
    scale = 1

    # fig, (ax1, ax2) = plt.subplots( 2, 1, figsize=(16/scale, 9/scale), subplot_kw={'projection': '3d'})
    fig, ax1 = plt.subplots(figsize=(16 / scale, 9 / scale), subplot_kw={'projection': '3d'})

    X, Y = np.meshgrid(dataX, dataY)
    Z = np.array(dataZ)
    # ground = np.zeros_like(X)

    # Plot a surface
    # Give the first plot only wireframes of the type y = c
    ax1.plot_wireframe(X, Y, Z, rstride=10, cstride=0)


    # Give the second plot only wireframes of the type x = c
    # ax2.plot_wireframe(X, Y, Z, rstride=0, cstride=10)
    # ax2.set_title("Row (y) stride set to 0")


    # Add labels
    ax1.set_title(descr.title)
    ax1.set_xlabel(descr.label_x)
    ax1.set_ylabel(descr.label_y)
    ax1.set_zlabel(descr.label_z)
    # ax.view_init(elev=90, azim=0)
    plt.tight_layout()

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300)  # Save as PNG file

    if plot_immediately:
        plt.show()

from matplotlib.colors import Normalize
def plot3Dplane_data(dataX, dataY, dataZ, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    # Create a figure and 3D axes
    scale = 1.5
    fig = plt.figure(figsize=(16/scale, 9/scale))
    ax = fig.add_subplot()

    X, Y = np.meshgrid(dataX, dataY)
    Z = np.array(dataZ)
    # ground = np.zeros_like(X)

    t = np.linspace(0, 2 * np.pi, 1024)
    data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

    # fig, ax = plt.subplots()

    im = ax.imshow(Z)
    ax.set_title('Pan on the colorbar to shift the color mapping\n'
                 'Zoom on the colorbar to scale the color mapping')

    fig.colorbar(im, ax=ax, label='Interactive colorbar')

    # nx, ny, nz = 8, 10, 5
    # data_xy = np.arange(ny * nx).reshape(ny, nx) + 15 * np.random.random((ny, nx))

    # print(type(data_xy))
    # print(data_xy[0])

    # Plot a surface
    # surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # surf = ax.plot_surface(X, Y, ground, facecolors=plt.cm.viridis((Z - Z.min()) / (Z.max() - Z.min())),
    #                        rstride=1, cstride=1, shade=False)
    # surf = ax.plot_surface(x_axis, time_scale, data_x_t, cmap='viridis')



    # cmap = 'viridis'
    # norm = None
    # if norm is None:
    #     norm = Normalize()
    # colors = plt.get_cmap(cmap)(norm(Z))
    # pos = 0
    # # ny, nx = data_xy.shape
    # # yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
    # zi = np.full_like(X, pos)
    #
    # ax.plot_surface(X, Y, zi, rstride=1, cstride=1, facecolors=colors, shade=False)




    # for i in range(0, len(dataY), 20):
    #     # y = y_base + z * 0.1  # Optionally modify the curve (e.g., add offset)
    #     ax.plot(dataX, ys=dataY[i], zs=dataZ, zdir='y', label=f'Curve at y={dataY[i]:.1f}')

    # Add color bar
    # fig.colorbar(surf, shrink=0.5, aspect=10)
    # mappable = plt.cm.ScalarMappable(cmap='viridis')
    # mappable.set_array(Z)
    # fig.colorbar(mappable, shrink=0.5, aspect=10, label="Color Scale")

    # Disable perspective by setting a large viewing distance
    # ax.dist = 10  # The larger the value, the less perspective distortion
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())



    # Add labels
    ax.set_title(descr.title)
    ax.set_xlabel(descr.label_x)
    ax.set_ylabel(descr.label_y)
    # ax.set_zlabel(descr.label_z)
    # ax.view_init(elev=90, azim=0)

    if save_to_file:
        # Save the plot to a file
        plt.savefig(file_name, dpi=300)  # Save as PNG file

    if plot_immediately:
        plt.show()


def plot_all_graphs():
    plt.show()