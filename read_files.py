import pyvista as pv
import matplotlib.pyplot as mt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import re
import numpy as np
import ploting
import unit_convert
from unit_input import particles_parameters, const_e, const_M_e, T_e, n_e, T_i, const_M_pr, n_i

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# inputs

# Load the VTK file
# folder = "../res_data/beam_14_quiet/"
# file_for_graph = "testMy01_rhoe2_0.vtk"
# file_path = folder + file_for_graph
#
# # parameters
# Lx = 1
# Lz = 0.1
# nx = 4096
# nz = 1
# num_of_files = 1000
# step = 10
# dt = 0.005
# z_level = 0
#
# graph_animation = False
# graph_3d = False
# plot_xtE = True
# save_gif = False
# gif_name = "beam_11_E"


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


def get_lin_x_data(L,n):
    axis = []
    for i in range(0, n - 1):
        axis.append(i * (L/n))

    return axis


# def update(frame):
#     print(f"frame = {frame}")
#     pd = meshes[frame].point_data[mesh.array_names[0]]
#     y = pd[col * row: row * (col + 1) - 1]
#     if PIC_data.is_vector():
#         yx = []
#         yy = []
#         yz = []
#         res = []
#         for i in y:
#             yx.append(i[0])
#             yy.append(i[1])
#             yz.append(i[2])
#
#             res.append(np.sqrt(np.pow(i[0],2) + np.pow(i[1],2) + np.pow(i[2],2)))
#
#         print(res)
#         line.set_data(x, res)
#     else:
#         print(y)
#         line.set_data(x, y)
#     return line,


def get_data_row(data, column, row):
    return data[column * row: row * (column + 1) - 1]


def create_file_names(folder, file_for_graph, num_of_files, step):
    """
    Generates a list of file paths based on a given template and step size.

    Args:
        folder (str): The folder where the files are located.
        file_for_graph (str): A sample file name with a numeric suffix and extension.
        num_of_files (int): The total number of files to consider.
        step (int): The step interval for file numbering.

    Returns:
        list: A list of file paths.
    """
    # Regular expression to extract the base name, number, and file extension
    pattern = r"^(.*)_(\d+)\.(\w+)$"

    match = re.match(pattern, file_for_graph)
    if not match:
        raise ValueError("The file name does not match the expected format: 'name_number.extension'")

    base_name, _, file_extension = match.groups()
    base_name_with_folder = f"{folder}{base_name}_"
    file_extension = f".{file_extension}"

    # Generate file paths using list comprehension
    file_paths = [
        f"{base_name_with_folder}{i}{file_extension}"
        for i in range(0, num_of_files, step)
    ]

    return file_paths


def read_data_name(file_path):
    mesh = pv.read(file_path)
    return mesh.point_data.keys()[0]


def read_file_data(file_path, grid_x, grid_z):
    """
    Reads a file, extracts mesh data, and retrieves specific rows of data.

    Args:
        file_path (str): Path to the file to be read.
        grid_x (int): Number of grid points along the x-axis.
        grid_z (int): Number of grid points along the z-axis.

    Returns:
        list or np.array: The extracted data for the specified row.
    """
    try:
        # Read the file using PyVista
        mesh = pv.read(file_path)

        # Get the point data from the first available array
        data = mesh.point_data[mesh.array_names[0]]

        # Extract the middle row of the grid based on z and x dimensions
        return get_data_row(data, int(grid_z / 2), int(grid_x))
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except KeyError as e:
        raise KeyError(f"Data array not found in file: {file_path}. Error: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {file_path}. Error: {e}")


def read_file_time_dataset(file_paths, grid_x, grid_z):
    """
    Reads and processes data from a series of files over a time range.

    Args:
        file_paths (list of str): List of file paths to read data from.
        grid_x (int): The number of grid points along the x-axis.
        grid_z (int): The number of grid points along the z-axis.

    Returns:
        list: A list of datasets, one for each file.
    """
    if not isinstance(file_paths, list) or not all(isinstance(path, str) for path in file_paths):
        raise ValueError("file_paths must be a list of strings representing file paths.")

    return [read_file_data(path, grid_x, grid_z) for path in file_paths]


def get_point_through_time(data, n, vector, dim_val="x"):
    """
    Extracts a list of values through time for a specific point index.

    Args:
        data (list): A list of time-step data.
        n (int): The point index to extract data for.
        vector (bool): Indicates whether the data is a vector (True) or scalar (False).
        dim_val (str): The dimension to extract ("x", "y", "z") if vector is True.

    Returns:
        list: A list of float values extracted across time for the given point index.
    """
    if not vector:
        # Scalar data extraction
        return [float(d[n]) for d in data]

    # Map dim_val to the appropriate index
    dim_map = {"x": 0, "y": 1, "z": 2}
    dim_idx = dim_map.get(dim_val, 0)  # Default to 0 if dim_val is invalid

    # Vector data extraction
    return [float(d[n][dim_idx]) for d in data]


def get_point_through_len(data, t, vector, dim_val="x"):
    """
    Extracts a list of values based on whether the input data is a scalar or a vector.

    Args:
        data (list): A list of data points.
        t (int): The index of the time step.
        vector (bool): Indicates whether the data is a vector (True) or scalar (False).
        dim_val (str): The dimension to extract ("x", "y", "z") if vector is True.

    Returns:
        list: A list of float values extracted based on the input parameters.
    """
    if not vector:
        # Use list comprehension for efficiency and readability
        return [float(d) for d in data[t]]

    # Map dim_val to the appropriate index
    dim_map = {"x": 0, "y": 1, "z": 2}
    dim_idx = dim_map.get(dim_val, 0)  # Default to 0 if dim_val is invalid

    # Extract the specified dimension using list comprehension
    return [float(d[dim_idx]) for d in data[t]]


def get_xz_data_vector(data, time_range, dim_val="x"):
    """
    Extracts a vector of data over a specified time range for a given dimension.

    Args:
        data (list): The input data containing vectors over time.
        time_range (list): A list of time indices to extract data for.
        dim_val (str): The dimension to extract ("x", "y", or "z"). Default is "x".

    Returns:
        list: A list of vectors for the specified dimension across the given time range.
    """
    # Map dimension to index
    dim_mapping = {"x": 0, "y": 1, "z": 2}
    if dim_val not in dim_mapping:
        raise ValueError(f"Invalid dimension value '{dim_val}'. Expected one of {list(dim_mapping.keys())}.")

    # Extract data for the specified time range and dimension
    return [get_point_through_len(data, i, True, dim_val) for i in range(0, len(time_range))]


def max_value(inputlist):
    return max([sublist[-1] for sublist in inputlist])


def min_value(inputlist):
    return min([sublist[-1] for sublist in inputlist])

class ReadFilesData:
    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz):
        self.file_paths = create_file_names(folder, file_for_graph, num_of_files, step)
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt

        self.data_x_t = read_file_time_dataset(self.file_paths, nx, nz)
        # self.vector = not isinstance(data_x_t[0], np.floating)

    def is_vector(self):
        print(self.data_x_t[0][0])
        print(np.floating)
        return not isinstance(self.data_x_t[0][0], np.floating)

    def get_2D_data(self, vector_component="x"):
        if self.is_vector():
            return get_xz_data_vector(self.data_x_t, self.get_time_data(), vector_component)
        else:
            return self.data_x_t

    def get_data_name(self):
        return read_data_name(self.file_paths[0])

    def get_x_data(self, L):
        # print(f"class L = {L}")
        # print(f"class n = {n}")
        # print(f"class x data len = {len(self.get_point_through_len(0))}")

        n = len(self.get_point_through_len(0))+1
        return [i * (L/n) for i in range(0, n - 1)]

    def get_time_data(self):
        cycles = list(range(0, self.num_of_files, self.step))
        return unit_convert.rescale_list(cycles, self.dt)

    def get_file_paths(self):
        return self.file_paths

    def get_point_through_len(self, t, dim_val="x"):
        """
            Extracts a list of values based on whether the input data is a scalar or a vector.

            Args:
                data (list): A list of data points.
                t (int): The index of the time step.
                vector (bool): Indicates whether the data is a vector (True) or scalar (False).
                dim_val (str): The dimension to extract ("x", "y", "z") if vector is True.

            Returns:
                list: A list of float values extracted based on the input parameters.
            """
        data = self.get_2D_data(dim_val)
        return [float(d) for d in data[t]]

        # if not vector:
        #     # Use list comprehension for efficiency and readability
        #     return [float(d) for d in data[t]]
        #
        # # Map dim_val to the appropriate index
        # dim_map = {"x": 0, "y": 1, "z": 2}
        # dim_idx = dim_map.get(dim_val, 0)  # Default to 0 if dim_val is invalid
        #
        # print(f"dim_idx = {dim_idx}")
        # print(f"data[t][d] = {data[t][1]}")
        # # Extract the specified dimension using list comprehension
        # return [float(d[dim_idx]) for d in data[t]]

    def get_point_through_time(self, n, dim_val="x"):
        """
        Extracts a list of values through time for a specific point index.

        Args:
            data (list): A list of time-step data.
            n (int): The point index to extract data for.
            vector (bool): Indicates whether the data is a vector (True) or scalar (False).
            dim_val (str): The dimension to extract ("x", "y", "z") if vector is True.

        Returns:
            list: A list of float values extracted across time for the given point index.
        """
        data = self.get_2D_data(dim_val)
        return [float(d[n]) for d in data]
        # if not vector:
        #     # Scalar data extraction
        #     return [float(d[n]) for d in data]
        #
        # # Map dim_val to the appropriate index
        # dim_map = {"x": 0, "y": 1, "z": 2}
        # dim_idx = dim_map.get(dim_val, 0)  # Default to 0 if dim_val is invalid
        #
        # # Vector data extraction
        # return [float(d[n][dim_idx]) for d in data]

    def print_mesh_data(self):
        mesh = pv.read(self.file_paths[0])
        point_data = mesh.point_data[mesh.array_names[0]]
        # Access points and cells
        print("Num. of points:", len(mesh.points))
        print("Cells:", mesh.cell)
        print("Point data: ", mesh.point_data.keys())
        print("Cells data", mesh.cell_data.keys())
        print("Simulation bounds: " + str(mesh.bounds))
        print("Central of simulation: " + str(mesh.center))
        print("Array names, chosen first name to plot: " + str(mesh.array_names))
        print("Values on grid")
        print("Data length: " + str(len(point_data)))


# PIC_data = ReadFilesData(folder, file_for_graph, num_of_files, step, dt)
# print(f" Is vector? {PIC_data.is_vector()}")
# # file_paths = create_file_names(folder, file_for_graph, num_of_files, step)
# # print(file_paths)
# data_x_t = PIC_data.get_2D_data("x")
# data_name = PIC_data.get_data_name()
# axis_time = PIC_data.get_time_data()
# axis_x = PIC_data.get_x_data(Lx)
#
#
#
#
# col = z_level
# row = nx
#
# # graph data
#
# z_axis = np.array(get_lin_x_data(Lz, nz))
#
# print(f"data size: m x n : {nx} x {nz} = {nz * nx}")
#
#
#
# # units
# electron = particles_parameters(const_e, T_e, const_M_e, n_e*10**6)
# ion = particles_parameters(const_e, T_i, const_M_pr, n_i*10**6)
#
# axis_x_SI = unit_convert.rescale_list(axis_x, ion.get_ion_skin_depth())
# axis_time_SI = unit_convert.rescale_list(axis_time, 1/ion.get_plasma_frequency())
# print(f"time {4163 / ion.get_plasma_frequency()}")
# print(f"time step: {dt} -> {(1/ion.get_plasma_frequency()) * dt} s")
# aa = 50
# print(f"data availeble for every: {axis_time_SI[aa] - axis_time_SI[aa-1]} s")
#
# # print(axis_x[-1])
# # print(axis_x_SI[-1])
# print(f"data type {max_value(data_x_t)}")
#
# if graph_3d:
#     mesh = pv.read(file_path)
#     point_data = mesh.point_data[mesh.array_names[0]]
#     meshes = [pv.read(file) for file in PIC_data.get_file_paths()]
#     plotter = pv.Plotter()
#     if graph_animation:
#
#         if save_gif:
#             # animation plot data, save to gif
#
#             plotter.open_gif(gif_name + ".gif")  # Save as a GIF (optional)
#
#             for mesh in meshes:
#                 plotter.clear()  # Clear previous frame
#                 plotter.add_mesh(mesh, scalars=mesh.point_data.keys()[0], show_edges=False)  # Replace `YourScalarName` with scalar field
#                 plotter.camera_position = "xz"
#                 plotter.write_frame()  # Save frame to the GIF
#
#             plotter.close()  # Close the GIF writer
#
#         else:
#             # plotter1 = pv.Plotter()
#             for mesh in meshes:
#                 plotter.clear()
#                 plotter.add_mesh(mesh, scalars=mesh.point_data.keys()[0], show_edges=False)
#                 plotter.camera_position = "xz"
#                 plotter.show(auto_close=False, interactive_update=True)
#
#     else:
#         print(mesh.point_data.keys()[0])
#         # simply plot data
#         plotter.add_mesh(mesh, scalars=mesh.point_data.keys()[0], show_edges=False, color='white')
#         plotter.camera_position = "xz"
#         plotter.add_title(f"Data for {mesh.point_data.keys()[0]}")
#         plotter.show_axes()
#         plotter.show()
#
#         # Create a figure and 3D axes
#         # fig = mt.figure()
#         # ax = fig.add_subplot(111, projection='3d')
#
#         # Generate example data
#         # x = np.linspace(-5, 5, 100)
#         # y = np.linspace(-5, 5, 100)
#         # X, Y = np.meshgrid(x, y)
#         # Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
#
#         # X, Y = np.meshgrid(axis_x, axis_time)
#         # Z = np.array(PIC_data.get_2D_data())
#
#
#         # print(f"len X: {len(X)}")
#         # print(f"len X[0]: {len(X[0])}")
#         # print(f"len Y: {len(Y)}")
#         # print(f"len Y[0]: {len(Y[0])}")
#         # print(f"len Z: {len(Z)}")
#         # print(f"len Z[0]: {len(Z[0])}")
#         #
#         #
#         #
#         # # Plot a surface
#         # surf = ax.plot_surface(X, Y, Z, cmap='viridis')
#         # surf = ax.plot_surface(x_axis, time_scale, data_x_t, cmap='viridis')
#
#         # Add color bar
#         # fig.colorbar(surf)
#         #
#         # # Add labels
#         # ax.set_title("3D Surface Plot")
#         # ax.set_xlabel("X Axis")
#         # ax.set_ylabel("Y Axis")
#         # ax.set_zlabel("Z Axis")
#         #
#         # # Show the plot
#         # mt.show()
#
# elif plot_xtE:
#     # val1 = get_point_through_len(data_x_t, len(axis_time)-1, PIC_data.is_vector(), "x")
#     x_level = 1
#     val1 = PIC_data.get_point_through_len(x_level, "x")
#     descr1 = ploting.PlotDescription(f"data {data_name} for time {axis_time[x_level]}", "length [isd]", "el. intensity [ ]")
#
#     # ploting.plot_data(axis_x, val1, descr1, False, "plot_E(x)")
#
#     descr11 = ploting.PlotDescription(f"data {data_name} for t = {round(axis_time_SI[x_level], 3)} s", "length [m]", "el. intensity [ ]")
#     descr11.set_ylim(min_value(data_x_t)*1.1, max_value(data_x_t)*1.1)
#     ploting.plot_data(axis_x_SI, val1, descr11, True, "plot_E_SI(x)")
#
#     # val2 = get_point_through_time(data_x_t, 4094, PIC_data.is_vector(), "x")
#     t_level = 4094
#     val2 = PIC_data.get_point_through_time(t_level, "x")
#     print(f"len val2: {len(val2)}")
#     print(f"len x_axis: {len(axis_x)}")
#     descr2 = ploting.PlotDescription(f"data {data_name} for len {axis_x[4094]}", "time [pf]", "el. intensity [ ]")
#     # ploting.plot_data(axis_time, val2, descr2, True, "plot3D_E(t)")
#
#     descr22 = ploting.PlotDescription(f"data {data_name} for x = {round(axis_x_SI[t_level],1)} m", "time [s]", "el. intensity [ ]")
#     descr22.set_ylim(min_value(data_x_t)*1.1, max_value(data_x_t)*1.1)
#     ploting.plot_data(axis_time_SI, val2, descr22, True, "plot_E_SI(t)")
#
#     descr3D = ploting.PlotDescription(f"time development for {data_name}","length [m]","time [s]","el. intensity [ ]")
#     ploting.plot3D_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D,True, "plot3D_E(x,t)")
#
#     ploting.plot3Dwire_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D, False, "plot3D_E(x,t)")
#     ploting.plot3Dplane_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D, False, "plot3D_E(x,t)")
#
#     ploting.plot_all_graphs()
#
# else:
#     mesh = pv.read(file_path)
#     point_data = mesh.point_data[mesh.array_names[0]]
#     print(f"data len: {len(point_data)}")
#     if graph_animation:
#         x = axis_x
#         # Set up the figure and axis
#         fig, ax = mt.subplots()
#         ax.set_xlim(0, axis_x[-1])
#         ax.set_ylim(point_data.min()*1.1, point_data.max()*1.1)
#         line, = ax.plot([], [], lw=2)
#
#         # Update function
#
#         # Create animation
#         anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
#
#         # Display
#         mt.show()
#
#     else:
#         if PIC_data.is_vector():
#             mt.plot(axis_x, get_data_row(point_data, col, row), linestyle='-', label=["X", "Y", "Z"])
#             mt.legend()
#         else:
#             mt.plot(axis_x, get_data_row(point_data, col, row), linestyle='-')
#
#         # Add labels and title
#         mt.xlabel("length (ion skin depth)")
#         mt.ylabel("value " + mesh.point_data.keys()[0])
#         mt.title(f"X data for {mesh.point_data.keys()[0]}, Z-level is {z_axis[col]} isd")
#           # Add legend for clarity
#
#         # Show the plot
#         mt.grid(True)  # Optional: Add gridlines
#         mt.show()
#
#
#
#
