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

def add_suffix(file_name, suffix):
    splited_name = file_name.split(".")
    return splited_name[0] + suffix + splited_name[1]

class ReadVTKFilesData:
    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz):
        """
                Initializes the ReadFilesData class.

                Args:
                    folder (str): The folder containing the files.
                    file_for_graph (str): The base name of the file for graphing.
                    num_of_files (int): Total number of files.
                    step (int): Step size for file loading.
                    dt (float): Time step for the simulation.
                    nx (int): Number of points in the x-direction.
                    nz (int): Number of points in the z-direction.
                """
        self.file_paths = create_file_names(folder, file_for_graph, num_of_files, step)
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt

        self.data_x_t = read_file_time_dataset(self.file_paths, nx, nz)
        self.data2D_x = None
        self.data2D_y = None
        self.data2D_z = None
        print("All files loaded...")

    def is_vector(self):
        """Checks if the data is vector data."""
        # return isinstance(self.data_x_t[0][0], (list, tuple))
        return not isinstance(self.data_x_t[0][0], np.floating)

    def preprocess_2D_data(self,  vector_component="x"):
        """
                    Extracts a 2D array of data over time for a specified vector component.

                    Args:
                        vector_component (str): The dimension to extract ("x", "y", or "z"). Default is "x".

                    Returns:
                        list: A 2D list of values for the specified component across all time steps.
                    """
        # Check if the data is vector data
        if not self.is_vector():
            return self.data_x_t  # Return scalar data directly

        # Map vector component to index
        dim_mapping = {"x": 0, "y": 1, "z": 2}
        if vector_component not in dim_mapping:
            raise ValueError(
                f"Invalid vector component '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")

        dim_idx = dim_mapping[vector_component]

        # Use a nested list comprehension to extract the specified vector component
        return [[float(d[dim_idx]) for d in time_step] for time_step in self.data_x_t]


    def get_2D_data(self, vector_component="x"):
        """
            Extracts a 2D array of data over time for a specified vector component.

            Args:
                vector_component (str): The dimension to extract ("x", "y", or "z"). Default is "x".

            Returns:
                list: A 2D list of values for the specified component across all time steps.
            """
        # Check if the data is vector data
        if not self.is_vector():
            return self.data_x_t  # Return scalar data directly

        # Map vector component to index
        dim_mapping = {"x": 0, "y": 1, "z": 2}
        if vector_component not in dim_mapping:
            raise ValueError(
                f"Invalid vector component '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")

        dim_idx = dim_mapping[vector_component]

        if dim_idx == 0:
            if self.data2D_x == None:
                print("data x loading...")
                self.data2D_x = self.preprocess_2D_data("x")
            return self.data2D_x

        if dim_idx == 1:
            if self.data2D_y == None:
                self.data2D_y = self.preprocess_2D_data("y")
            return self.data2D_y

        if dim_idx == 2:
            if self.data2D_z == None:
                self.data2D_z = self.preprocess_2D_data("z")
            return self.data2D_z

        # Use a nested list comprehension to extract the specified vector component
        # return [[float(d[dim_idx]) for d in time_step] for time_step in self.data_x_t]
        # if self.is_vector():
    #         """
    #             Extracts a vector of data over a specified time range for a given dimension.
    #
    #             Args:
    #                 data (list): The input data containing vectors over time.
    #                 time_range (list): A list of time indices to extract data for.
    #                 dim_val (str): The dimension to extract ("x", "y", or "z"). Default is "x".
    #
    #             Returns:
    #                 list: A list of vectors for the specified dimension across the given time range.
    #             """
    #         # Map dimension to index
    #         dim_mapping = {"x": 0, "y": 1, "z": 2}
    #         if vector_component not in dim_mapping:
    #             raise ValueError(f"Invalid dimension value '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")
    #
    #         # Extract data for the specified time range and dimension
    #         aaa = []
    #         for i in range(0, len(self.get_time_data())):
    #
    #             if not self.is_vector():
    #                 # Use list comprehension for efficiency and readability
    #                 return [float(d) for d in self.data_x_t[i]]
    #
    #                 # Map dim_val to the appropriate index
    #             dim_map = {"x": 0, "y": 1, "z": 2}
    #             dim_idx = dim_map.get(vector_component, 0)  # Default to 0 if dim_val is invalid
    #
    #             # Extract the specified dimension using list comprehension
    #             bbb = []
    #             for d in self.data_x_t[i]:
    #                 bbb.append(float(d[dim_idx]))
    #             # return [float(d[dim_idx]) for d in data[t]]
    #             rrr = bbb
    #             # rrr = get_point_through_len(self.data_x_t, i, True, vector_component)
    #             aaa.append(rrr)
    #
    #
    #         return aaa
    #     if self.is_vector():
    #         return get_xz_data_vector(self.data_x_t, self.get_time_data(), vector_component)
    #     else:
    #         return self.data_x_t

    def get_data_name(self):
        """Gets the name of the data from the first file."""
        return read_data_name(self.file_paths[0])

    def get_x_data(self, L):
        """
                Generates the x-axis values based on the domain length.

                Args:
                    L (float): The length of the domain.

                Returns:
                    list: x-axis values.
                """
        n = len(self.get_point_through_len(0)) + 1
        return [i * (L / n) for i in range(n - 1)]


    def get_time_data(self):
        """
                Generates the time data based on the time step and total cycles.

                Returns:
                    list: A rescaled list of time values.
                """
        cycles = list(range(0, self.num_of_files, self.step))
        return unit_convert.rescale_list(cycles, self.dt)

    def get_file_paths(self):
        """Returns the list of file paths."""
        return self.file_paths

    def get_point_through_len(self, t, dim_val="x"):
        """
        Extracts data for a specific time step across all points.

        Args:
            t (int): The time step index.
            dim_val (str): The dimension to extract ("x", "y", "z").

        Returns:
            list: A list of float values for the specified time step and dimension.
        """
        data = self.get_2D_data(dim_val)
        return [float(d) for d in data[t]]

    def get_point_through_time(self, n, dim_val="x"):
        """
        Extracts data over time for a specific point.

        Args:
            n (int): The point index.
            dim_val (str): The dimension to extract ("x", "y", "z").

        Returns:
            list: A list of float values across time for the specified point and dimension.
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


class ReadHDFFilesData(ReadVTKFilesData):
    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz):
        """
                Initializes the ReadFilesData class.

                Args:
                    folder (str): The folder containing the files.
                    file_for_graph (str): The base name of the file for graphing.
                    num_of_files (int): Total number of files.
                    step (int): Step size for file loading.
                    dt (float): Time step for the simulation.
                    nx (int): Number of points in the x-direction.
                    nz (int): Number of points in the z-direction.
                """
        self.file_paths = create_file_names(folder, file_for_graph, num_of_files, step)
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt

        self.data_x_t = read_file_time_dataset(self.file_paths, nx, nz)
        self.data2D_x = None
        self.data2D_y = None
        self.data2D_z = None
        print("All files loaded...")


    def read_hdf_file(self):
        pass


if __name__ == '__main__':
    file = ""
