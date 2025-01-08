import pyvista as pv
import re
import numpy as np
import unit_convert


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
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
    return max(max(sublist) for sublist in inputlist)
    # return max([sublist[-1] for sublist in inputlist])


def min_value(inputlist):
    return min(min(sublist) for sublist in inputlist)
    # return min([sublist[-1] for sublist in inputlist])

class ReadFilesData:
    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz):
        self.file_paths = create_file_names(folder, file_for_graph, num_of_files, step)
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt

        self.data_x_t = read_file_time_dataset(self.file_paths, nx, nz)
        # self.vector = not isinstance(data_x_t[0], np.floating)

    def is_vector(self):
        return not isinstance(self.data_x_t[0][0], np.floating)

    def get_2D_data(self, vector_component="x"):
        if self.is_vector():
            return get_xz_data_vector(self.data_x_t, self.get_time_data(), vector_component)
        else:
            return self.data_x_t

    def get_data_name(self):
        return read_data_name(self.file_paths[0])

    def get_x_data(self, L):
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


    # def get_global_min(self):
    #     return min(min(sublist) for sublist in self.get_2D_data())
    #
    # def get_global_max(self):
    #     return max(max(sublist) for sublist in self.get_2D_data())

