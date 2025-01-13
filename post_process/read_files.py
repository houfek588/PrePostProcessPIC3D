import pyvista as pv
import re
import numpy as np
import unit_convert
import h5py



# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------






def get_data_row(data, column, row):
    return data[column * row: row * (column + 1) - 1]





# def read_data_name(file_path):
#     mesh = pv.read(file_path)
#     return mesh.point_data.keys()[0]


# def read_file_data(file_path, grid_x, grid_z):
#     """
#     Reads a file, extracts mesh data, and retrieves specific rows of data.
#
#     Args:
#         file_path (str): Path to the file to be read.
#         grid_x (int): Number of grid points along the x-axis.
#         grid_z (int): Number of grid points along the z-axis.
#
#     Returns:
#         list or np.array: The extracted data for the specified row.
#     """
#     try:
#         # Read the file using PyVista
#         mesh = pv.read(file_path)
#
#         # Get the point data from the first available array
#         data = mesh.point_data[mesh.array_names[0]]
#
#         # Extract the middle row of the grid based on z and x dimensions
#         return get_data_row(data, int(grid_z / 2), int(grid_x))
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File not found: {file_path}")
#     except KeyError as e:
#         raise KeyError(f"Data array not found in file: {file_path}. Error: {e}")
#     except Exception as e:
#         raise RuntimeError(f"An error occurred while reading the file: {file_path}. Error: {e}")


def max_value(inputlist):
    return max(max(sublist) for sublist in inputlist)
    # return max([sublist[-1] for sublist in inputlist])


def min_value(inputlist):
    return min(min(sublist) for sublist in inputlist)
    # return min([sublist[-1] for sublist in inputlist])

def add_suffix(file_name, suffix):
    splited_name = file_name.split(".")
    return splited_name[0] + suffix + splited_name[1]


def avg_2n2_matrix(input_matrix):
    total_sum = sum(sum(row) for row in input_matrix)  # Sum all elements
    total_count = sum(len(row) for row in input_matrix)  # Count all elements
    return total_sum / total_count if total_count > 0 else 0  # Avoid division by zero

def get_vector_variables():
    return ["E", "B"]


def parse_text_and_number(input_string):
    """
    Parses a string into its text and numeric components.

    Args:
        input_string (str): The string to parse (e.g., "restart1").

    Returns:
        tuple: A tuple of the text and the number (e.g., ("restart", 1)).
               Returns (None, None) if parsing fails.
    """
    # pattern = r"([a-zA-Z]+)(\d{1,3})$"  # Match letters followed by 1-3 digits
    pattern = r"^(cycle)_(\d+)$"
    match = re.match(pattern, input_string)

    if match:
        text_part = match.group(1)
        number_part = int(match.group(2))

        return text_part, number_part
    else:
        return None, None  # Handle invalid input gracefully

class ReadVTKFilesData:
    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz, variable="E", axis="x"):
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
        self.file_paths = self.create_file_names(folder, file_for_graph, num_of_files, step)
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt
        self.variable = variable

        if not self.is_vector():
            # return self.data_x_t  # Return scalar data directly

            self.data_x_t = self.read_file_time_dataset(self.file_paths, nx, nz)
        else:
            raw_data = self.read_file_time_dataset(self.file_paths, nx, nz)
            self.data_x_t = self.preprocess_2D_data(raw_data, axis)

        # self.data2D_x = None
        # self.data2D_y = None
        # self.data2D_z = None

        self.len_data = None
        self.time_data = None
        print("All files loaded...")

    def is_vector(self):
        """Checks if the data is vector data."""
        if self.variable in get_vector_variables():
            return True
        return False

    def create_file_names(self, folder, file, num_of_files, step):
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

        match = re.match(pattern, file)
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

    def read_file_time_dataset(self, file_paths, grid_x, grid_z):
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

        grow_data = []
        for path in file_paths:
            try:
                # Read the file using PyVista
                mesh = pv.read(path)

                # Get the point data from the first available array
                data = mesh.point_data[mesh.array_names[0]]

                # Extract the middle row of the grid based on z and x dimensions
                row_data = get_data_row(data, int(grid_z / 2), int(grid_x))
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {path}")
            except KeyError as e:
                raise KeyError(f"Data array not found in file: {path}. Error: {e}")
            except Exception as e:
                raise RuntimeError(f"An error occurred while reading the file: {path}. Error: {e}")

            # rr = self.read_file_data(path, grid_x, grid_z)
            grow_data.append(row_data)
        return grow_data
        # return [self.read_file_data(path, grid_x, grid_z) for path in file_paths]

    def preprocess_2D_data(self, raw_data, vector_component="x"):
        """
                    Extracts a 2D array of data over time for a specified vector component.

                    Args:
                        vector_component (str): The dimension to extract ("x", "y", or "z"). Default is "x".

                    Returns:
                        list: A 2D list of values for the specified component across all time steps.
                    """
        # Check if the data is vector data
        # if not self.is_vector():
        #     return raw_data  # Return scalar data directly

        # Map vector component to index
        dim_mapping = {"x": 0, "y": 1, "z": 2}
        if vector_component not in dim_mapping:
            raise ValueError(
                f"Invalid vector component '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")

        dim_idx = dim_mapping[vector_component]

        # Use a nested list comprehension to extract the specified vector component
        return [[float(d[dim_idx]) for d in time_step] for time_step in raw_data]

    def get_field1D_len(self, t):
        """
        Extracts data for a specific time step across all points.

        Args:
            t (int): The time step index.
            dim_val (str): The dimension to extract ("x", "y", "z").

        Returns:
            list: A list of float values for the specified time step and dimension.
        """
        # data = self.get_2D_data(dim_val)
        # return [float(d) for d in data[t]]
        return self.data_x_t[t]

    def get_field1D_time(self, n):
        """
        Extracts data over time for a specific point.

        Args:
            n (int): The point index.
            dim_val (str): The dimension to extract ("x", "y", "z").

        Returns:
            list: A list of float values across time for the specified point and dimension.
        """
        # data = self.get_2D_data(dim_val)
        return [float(d[n]) for d in self.data_x_t]

    def get_2D_data(self):
        """
            Extracts a 2D array of data over time for a specified vector component.

            Args:
                vector_component (str): The dimension to extract ("x", "y", or "z"). Default is "x".

            Returns:
                list: A 2D list of values for the specified component across all time steps.
            """
        return self.data_x_t

        # # Check if the data is vector data
        # if not self.is_vector():
        #     return self.data_x_t  # Return scalar data directly
        #
        # # Map vector component to index
        # dim_mapping = {"x": 0, "y": 1, "z": 2}
        # if vector_component not in dim_mapping:
        #     raise ValueError(
        #         f"Invalid vector component '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")
        #
        # dim_idx = dim_mapping[vector_component]

        # if dim_idx == 0:
        #     if self.data2D_x == None:
        #         print("data x loading...")
        #         self.data2D_x = self.preprocess_2D_data("x")
        #     return self.data2D_x
        #
        # if dim_idx == 1:
        #     if self.data2D_y == None:
        #         self.data2D_y = self.preprocess_2D_data("y")
        #     return self.data2D_y
        #
        # if dim_idx == 2:
        #     if self.data2D_z == None:
        #         self.data2D_z = self.preprocess_2D_data("z")
        #     return self.data2D_z

        # Use a nested list comprehension to extract the specified vector component
        # return [[float(d[dim_idx]) for d in time_step] for time_step in self.data_x_t]
        # if self.is_vector():


    def get_data_name(self):
        """Gets the name of the data from the first file."""
        mesh = pv.read(self.file_paths[0])
        return mesh.point_data.keys()[0]


    def get_len_data(self, L):
        """
                Generates the x-axis values based on the domain length.

                Args:
                    L (float): The length of the domain.

                Returns:
                    list: x-axis values.
                """
        if self.len_data == None:
            n = len(self.get_field1D_len(0)) + 1
            self.len_data = [i * (L / n) for i in range(n - 1)]
        return self.len_data

    def get_time_data(self):
        """
                Generates the time data based on the time step and total cycles.

                Returns:
                    list: A rescaled list of time values.
                """
        if self.time_data == None:
            cycles = list(range(0, self.num_of_files, self.step))
            self.time_data = unit_convert.rescale_list(cycles, self.dt)

        return self.time_data

    def get_file_paths(self):
        """Returns the list of file paths."""
        return self.file_paths

    def __str__(self):
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


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

class ReadHDFFieldData(ReadVTKFilesData):
    def __init__(self, folder, file_for_graph, num_of_files, variable="E", axis="x"):
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
        self.file_paths = self.create_file_names(folder, file_for_graph, num_of_files)

        self.num_of_files = num_of_files
        self.variable = variable
        self.axis = axis
        self.len_data = None
        self.time_data = None

        self.data_x_t = self.read_field_2D()
        print("All files loaded...")

    def create_file_names(self, folder, file, num_of_files):

        # split input file name to parsing
        split_names = file.split(".")
        name = split_names[0]
        type = split_names[1]
        pattern = r"([a-zA-Z]+)(\d{1,3})$"  # Match letters followed by 1-3 digits
        match = re.match(pattern, name)

        # check if file name sets to parsing
        if match:
            text_part = match.group(1)
            number_part = int(match.group(2))
            # return text_part, number_part
        else:
            raise Exception("Match error")
            # return None, None  # Handle invalid input gracefully

        # paths = []
        # for i in range(0,n):
        #     paths.append(folder + text_part + str(i) + "." + f[1])
        #
        # print(paths)
        # return paths
        return [folder + text_part + str(i) + "." + type for i in range(0, num_of_files)]

    def read_field1D_len(self, hdf_files, var, cycle_key):
        var_data = []
        for file_path in hdf_files:
            # print(f"Processing file: {file_path}")
            with h5py.File(file_path, "r") as hdf:
                # Validate keys
                if "fields" not in hdf or var not in hdf["fields"] or cycle_key not in hdf["fields"][var]:
                    raise KeyError(f"Missing required keys in file {file_path}")

                cycle_data = hdf["fields"][var][cycle_key]

                # Process each dataset in the cycle
                for c in cycle_data:
                    var_data.append(avg_2n2_matrix(c))
                var_data.pop(-1)


        # check length data and create it
        if self.len_data == None:
            self.len_data = range(0, len(var_data))

        return var_data

    def read_field_2D(self):
        if self.is_vector():
            col_name = self.variable + self.axis
        else:
            col_name = self.variable

        # keys = []
        with h5py.File(self.file_paths[0], "r") as hdf:
            # for k in hdf["fields"]["Ex"].keys():
            #     keys.append(k)
            keys = [k for k in hdf["fields"][col_name].keys()]

        sort_cycle_name = sorted(keys, key=lambda x: parse_text_and_number(x)[1])

        # check time data and create it
        if self.time_data == None:
            self.create_time_data(sort_cycle_name)

        # return sorted data via time
        return [self.read_field1D_len(self.file_paths, col_name, c) for c in sort_cycle_name]

    def get_field1D_len(self, time):
        return self.data_x_t[time]

    def get_field1D_time(self, length):
        # time_data = []
        # for t in self.data_x_t:
        #     time_data.append(t[length])
        # return time_data
        return [float(t[length]) for t in self.data_x_t]

    def create_time_data(self, sorted_cycles):

        # print(sorted_cycles)
        # numbers = []
        # for i in sorted_cycles:
        #     numbers.append(parse_text_and_number(i)[1])
        #
        # self.time_data = numbers
        self.time_data = [parse_text_and_number(i)[1] for i in sorted_cycles]
        print("time data created")

    def get_time_data(self):
        return self.time_data

    def get_len_data(self):
        return self.len_data

    def get_2D_data(self):
        return self.data_x_t

    # def get_file_paths(self):
    #     return self.file_paths


class ReadHDFParticleData(ReadHDFFieldData):

    def read_field_2D(self):

        species = self.variable
        var = self.axis

        # keys = []
        with h5py.File(self.file_paths[0], "r") as hdf:
            # for k in hdf["fields"]["Ex"].keys():
            #     keys.append(k)
            keys = [k for k in hdf["particles"][species][var].keys()]

        sort_cycle_name = sorted(keys, key=lambda x: parse_text_and_number(x)[1])

        print(sort_cycle_name)
        # check time data and create it
        if self.time_data == None:
            self.create_time_data(sort_cycle_name)

        # return sorted data via time
        return [self.read_field1D_len(self.file_paths, species, var, c) for c in sort_cycle_name]

    def read_field1D_len(self, hdf_files, species, var, cycle_key):
        var_data = []
        for file_path in hdf_files:
            # print(f"Processing file: {file_path}")
            with h5py.File(file_path, "r") as hdf:
                # Validate keys
                if "particles" not in hdf or species not in hdf["particles"] or var in hdf["particles"][species] or cycle_key not in hdf["particles"][species][var]:
                    # raise KeyError(f"Missing required keys in file {file_path}")
                    pass

                cycle_data = hdf["particles"][species][var][cycle_key]

                # Process each dataset in the cycle
                for c in cycle_data:
                    var_data.append(c)
                # var_data.pop(-1)


        # check length data and create it
        if self.len_data == None:
            self.len_data = range(0, len(var_data))

        return var_data

    def get_particle1D_len(self, time):
        return super().get_field1D_len(time)

    def get_particle1D_time(self, length):
        return super().get_field1D_time(length)




if __name__ == '__main__':
    import ploting


    file_path = "../../res_data_vth/beam01_drftB/data_hdf5/restart1.hdf"

    paths = ["../../res_data_vth/beam01_drftB/data_hdf5/restart0.hdf",
             "../../res_data_vth/beam01_drftB/data_hdf5/restart1.hdf"]

    # disc_data = {"fields": }
    # h_file = ReadHDFFieldData("../../res_data_vth/beam01_drftB/data_hdf5/", "restart1.hdf", 32, "E", "x")

    h_file = ReadHDFParticleData("../../res_data_vth/beam01_drftB/data_hdf5/", "restart0.hdf", 32, "species_0", "u")

    # Open the HDF5 file
    # file_path = "example.h5"

    # dat = []
    # for p in paths:
    #
    #
    #     with h5py.File(p, "r") as hdf:
    #
    #
    #         res1 = read_field(hdf,"Ex","cycle_0")
    #
    #     dat.append(res1)
    # res1 = h_file.read_field1D_len(paths, "Ex", "cycle_0")
    # res1 = h_file.read_field_2D()
    # print(f"celk len: {len(res1)}")
    # print(res1[126:132])

    descr3D = ploting.PlotDescription(f"Time development through space for Ex", "length [db]",
                                      "time [1/Om_pi]",
                                      "Ex")
    # descr3D.set_ylim(min_value(data_x_t), max_value(data_x_t))
    x = h_file.get_len_data()
    x_t = h_file.get_2D_data()

    print(f"x len: {len(x)}")
    print("2D lengths:")
    print(f"time len: {len(x_t)}; x len: {len(x_t[0])}")

    # print(x[0])
    ploting.plot3Dplane_data(x, h_file.get_time_data(), x_t, descr3D, False, "vv")

    ploting.plot_all_graphs()

