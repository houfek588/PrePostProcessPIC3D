import pyvista as pv
import re
import numpy as np
import unit_convert
import h5py
import csv

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------


def get_data_row(data, column, row):
    return data[column * row: row * (column + 1) - 1]


def max_value(inputlist):
    return max(max(sublist) for sublist in inputlist)


def min_value(inputlist):
    return min(min(sublist) for sublist in inputlist)


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
            n = len(self.get_field1D_len(0))
            # self.len_data = [i * (L / n) for i in range(n - 1)]
            self.len_data = range(0, n)
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

        self.folder = folder
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

    def get_data_name(self):
        return self.variable
    # def get_file_paths(self):
    #     return self.file_paths

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------


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

        sett = ReadHDFSettings(self.folder + "settings.hdf")
        part_in_cel = sett.get_num_part_in_cell(self.variable, self.axis)

        # check length data and create it
        if self.len_data == None:
            raw = range(0, len(var_data))
            self.len_data = unit_convert.rescale_list(raw, 1/part_in_cel)

        return var_data

    def get_particle1D_len(self, time):
        return super().get_field1D_len(time)

    def get_particle1D_time(self, length):
        return super().get_field1D_time(length)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

class ReadHDFSettings:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_data = {}
        # self.load_file()
        self.dict_dir = {"x": "x",
                         "y": "y",
                         "z": "z",
                         "u": "x",
                         "v": "y",
                         "w": "z",
                         "q": "x"}

    def load_file(self):
        try:
            self.file_data = []
            with h5py.File(self.file_name, "r") as hdf:
                # Validate the "collective" key
                if "collective" not in hdf:
                    raise KeyError(f"The key 'collective' is missing in the file {self.file_name}")

                # Assign the data to the instance variable
                for i in hdf["collective"].keys():
                    print(f"i: {i}, type: {type(i)}")
                    print(hdf["collective"][i][0])
                    self.file_data.append(hdf["collective"][i][0])
                    # self.file_data[i] = hdf["collective"][i][0]

                print(f"File '{self.file_name}' loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{self.file_name}' was not found.")
        except KeyError as e:
            raise KeyError(f"Key error while loading file '{self.file_name}': {e}")
        except Exception as e:
            raise Exception(f"An error occurred while loading file '{self.file_name}': {e}")


    def check_available_dir(self, direction):
        # available = {"x", "y", "z", "u", "v", "w"}
        available = list(self.dict_dir.keys())


        # Validate the direction input
        if direction not in available:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {available}.")
        return True

    def get_num_cells(self, direction="x"):
        """
            Retrieves the number of cells for a specified direction.

            Args:
                direction (str): The direction to retrieve ("x", "y", or "z"). Default is "x".

            Returns:
                int: The number of cells in the specified direction.

            Raises:
                ValueError: If the specified direction is not valid.
            """
        available = {"x", "y", "z"}

        # Validate the direction input
        if direction not in available:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {available}.")

        # Construct the key and return the value
        cell_key = f"N{direction}c"

        with h5py.File(self.file_name, "r") as hdf:
            return hdf["collective"][cell_key][0]

    def get_box_size(self, direction="x"):
        """
            Retrieves the number of cells for a specified direction.

            Args:
                direction (str): The direction to retrieve ("x", "y", or "z"). Default is "x".

            Returns:
                int: The number of cells in the specified direction.

            Raises:
                ValueError: If the specified direction is not valid.
            """
        # available = {"x", "y", "z"}
        #
        # # Validate the direction input
        # if direction not in available:
        #     raise ValueError(f"Invalid direction '{direction}'. Must be one of {available}.")

        if self.check_available_dir(direction):
            # Construct the key and return the value
            cell_key = f"L{self.dict_dir[direction]}"

            with h5py.File(self.file_name, "r") as hdf:
                return hdf["collective"][cell_key][0]

    def get_num_cycles(self):
        with h5py.File(self.file_name, "r") as hdf:
            return hdf["collective"]["Ncycles"][0]

    def get_num_part_in_cell(self, species, direction="x"):

        if self.check_available_dir(direction):
            cell_key = f"Npcel{self.dict_dir[direction]}"
            # spec_key = f"species_{num_species}"


            with h5py.File(self.file_name, "r") as hdf:
                return hdf["collective"][species][cell_key][0]


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

class ReadConsData:
    def __init__(self, folder, file_for_graph):
        path = folder + file_for_graph

        self.data = {}
        self.read_file(path)

        # for k in self.data.keys():
        #     print(k)
        #     print(self.data.get(k)[0])
        #     print(self.data.get(k)[1][0])

    def read_file(self, file_path):
        self.data = {
            "cycle": ["Cycle", []],
            "total_energy": ["", []],
            "momentum": ["", []],
            "e_energy": ["", []],
            "b_energy": ["", []],
            "k_energy": ["", []],
            "k_energy_spec": ["", []],
            "bulk_energy_spec": ["", []]
        }
        keys = [
            "cycle",
            None,  # Skip column 1
            "total_energy",
            "momentum",
            "e_energy",
            "b_energy",
            "k_energy",
            "k_energy_spec",
            "bulk_energy_spec"
        ]

        with open(file_path, mode="r") as file:
            reader = csv.reader(file, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:  # Header row
                    for key, value in zip(keys, range(0, len(row))):
                        # print(f"key = {key}, value = {value}, row_val = {row[value]}")
                        if key:
                            if row[value]:
                                self.data[key][0] = row[value+1]  # Assign header title
                            else:
                                self.data[key][0] = row[value]
                                # self.data[key][0] = "Cycle"

                            if key == "cycle":
                                self.data[key][0] = "Cycle"
                            elif key == "total_energy":
                                self.data[key][0] = row[value + 1]
                else:  # Data rows
                    for idx, key in enumerate(keys):
                        if key:
                            self.data[key][1].append(float(row[idx]))  # Append data

    def get_cycles(self):
        return self.data["cycle"]

    def get_e_energy(self):
        return self.data["e_energy"]



if __name__ == '__main__':
    import ploting


    # file_path = "../../res_data_vth/beam01_drftB/data_hdf5/restart1.hdf"
    #
    # paths = ["../../res_data_vth/beam01_drftB/data_hdf5/restart0.hdf",
    #          "../../res_data_vth/beam01_drftB/data_hdf5/restart1.hdf"]


    # h_file = ReadHDFFieldData("../../res_data_vth/beam01_drftB/data_hdf5/", "restart1.hdf", 32, "E", "x")
    # h_file = ReadHDFParticleData("../../res_data_vth/beam01_drftB/data_hdf5/", "restart0.hdf", 32, "species_0", "u")


    descr3D = ploting.PlotDescription(f"Time development through space for Ex", "length [db]",
                                      "time [1/Om_pi]",
                                      "Ex")

    # x = h_file.get_len_data()
    # x_t = h_file.get_2D_data()
    #
    # print(f"x len: {len(x)}")
    # print("2D lengths:")
    # print(f"time len: {len(x_t)}; x len: {len(x_t[0])}")
    #
    # # print(x[0])
    # ploting.plot3Dplane_data(x, h_file.get_time_data(), x_t, descr3D, False, "vv")

    # ploting.plot_all_graphs()

    # set = ReadHDFSettings("../../res_data_vth/beam01_drftB/data/settings.hdf")
    #
    # dir1 = "x"
    # dir2 = "y"
    # print(f"number of cell in {dir1} direction: {set.get_num_cells(dir1)}")
    # print(f"number of cell in {dir2} direction: {set.get_box_size(dir2)}")
    # print(f"number of cycles: {set.get_num_cycles()}")
    #
    # print(f"number of particles in cell: {set.get_num_part_in_cell(1)}")

    en = ReadConsData("../../res_data_vth/beam02/data/", "ConservedQuantities.txt")

    # print(en.get_cycles()[0])
    # print(en.get_cycles()[1])
    #
    # print(en.get_cycles()[0])
    # print(en.get_e_energy()[1])

    des = ploting.PlotDescription(f"Time development energy E", en.get_cycles()[0],
                                      en.get_e_energy()[0])

    ploting.plot_data(en.get_cycles()[1], en.get_e_energy()[1], des)
    ploting.plot_all_graphs()