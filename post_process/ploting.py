import matplotlib.pyplot as plt
import numpy as np

plot_immediately = False

class PlotDescription:
    def __init__(self, title="title", label_x="axis x", label_y="axis y", label_z="axis z"):
        self.title = title
        self.label_x = label_x
        self.label_y = label_y
        self.label_z = label_z

        self.y_max = 0
        self.y_min = 0

    def set_ylim(self, minimum, maximum):
        self.y_max = maximum
        self.y_min = minimum




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
            # plt.scatter(dataX, dataY[i], color='red', label="Data Points", zorder=3)
    else:
        plt.plot(dataX, dataY)
        # plt.scatter(dataX, dataY, color='red', label="Data Points", zorder=3)

    plt.xlabel(descr.label_x)
    plt.ylabel(descr.label_y)
    plt.title(descr.title)
    plt.grid(True)

    # Show data points


    if descr.y_max == 0 and descr.y_min == 0:
        print("y limits dont used")
        pass
    else:
        print(f"y limits ({descr.y_min}, {descr.y_max})")
        plt.ylim(descr.y_min, descr.y_max)

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

    # X, Y = np.meshgrid(dataX, dataY)
    # Z = np.array(dataZ)
    Z = dataZ
    # ground = np.zeros_like(X)

    t = np.linspace(0, 2 * np.pi, 1024)
    # data2d = np.sin(t)[:, np.newaxis] * np.cos(t)[np.newaxis, :]

    # fig, ax = plt.subplots()
    # plt.imshow(Z, aspect='auto', extent=(axis_x_SI[0], axis_x_SI[-1], axis_time_SIms[0], axis_time_SIms[-1]),
    #            cmap='viridis')
    # im = ax.imshow(Z, aspect='auto', extent=(dataX[0], dataX[-1], dataY[0], dataY[-1]),
    #            cmap='viridis', origin="lower", vmin=descr.y_min, vmax=descr.y_max)
    im = ax.imshow(Z, aspect='auto', extent=(dataX[0], dataX[-1], dataY[0], dataY[-1]),
                              cmap='viridis', origin="lower")
    # im = ax.imshow((Z), extent=(-10, 10, -10, 10), cmap='viridis')
    ax.set_title(descr.title)

    fig.colorbar(im, ax=ax, label=descr.label_z)




    # plt.subplot(1, 2, 2)
    # plt.title("FFT Result (Frequency Domain)")
    # # plt.imshow(np.log1p(magnitude), extent=(-10, 10, -10, 10), cmap='magma')
    # plt.imshow((magnitude), extent=(-10, 10, -10, 10), cmap='magma')
    # plt.colorbar(label="Log Magnitude")






    # nx, ny, nz = 8, 10, 5
    # data_xy = np.arange(ny * nx).reshape(ny, nx) + 15 * np.random.random((ny, nx))




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


def plot_fft(dataX, dataY, descr: PlotDescription, save_to_file: bool = False, file_name="plot.png"):
    aa = 50
    step = dataX[aa] - dataX[aa - 1]

    # Perform FFT
    fft_result = np.fft.fft(dataY)
    frequencies = np.fft.fftfreq(len(dataY), d=step)

    # Get magnitude spectrum (optional)
    magnitude = np.abs(fft_result)

    # Generate plot
    plot_data(frequencies[1:len(frequencies) // 2], magnitude[1:len(magnitude) // 2], descr, save_to_file,
                      file_name)

def plot_all_graphs():
    plt.show()