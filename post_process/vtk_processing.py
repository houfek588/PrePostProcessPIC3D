import json
import post_process.read_files as read
import os
import re
import numpy as np


def extract_number(filename):
    """
    Extracts a numeric value from a given filename.

    Parameters:
    filename (str): The filename from which to extract the number.

    Returns:
    int: The extracted number if found.
    float: Returns infinity if no number is found (ensures such files are sorted last).
    """

    # Use regex to find the last numeric sequence before the file extension
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)

    # Convert the extracted number to an integer, or return infinity if no match is found
    return int(match.group()) if match else float('inf')


def process_folder(folder, variable_name, extract_number):
    """
    Processes a folder to find and sort specific .vtk files based on a given variable name.

    Parameters:
    folder (str): The path to the folder containing files.
    variable_name (str): The variable name to filter files by.
    extract_number (function): A function that extracts a numeric value for sorting.

    Returns:
    list: A sorted list of filenames that contain the specified variable name.
    str: A message if the folder is empty.
    """

    # List all files in the specified folder
    files = os.listdir(folder)

    # Check if the folder is empty
    if not files:
        return "The source data folder is empty."

    # Filter for .vtk files only
    vtk_files = [f for f in files if f.endswith(".vtk")]

    # Further filter for files containing the specified variable name
    par_files = [f for f in vtk_files if variable_name in f]

    # Sort the filtered files based on extracted numerical values
    par_sorted_files = sorted(par_files, key=extract_number)

    # Print the number of files to be processed
    print(f"The source data folder has {len(par_sorted_files)} files to process...")

    # Return the sorted list of files
    return par_sorted_files


def read_step_size(type):
    # with open("parameters_vtk.json", "r") as file:
    #     parameters = json.load(file)
    # return parameters["sim_parameters"]["step"]
    with open("config.json", "r") as file:
        parameters = json.load(file)

    if type == "part":
        return parameters["ParticleOutputCycle"]
    else:
        return parameters["FieldOutputCycle"]


def result_analysis(set_data):
    """
        Analyze VTK result data and save a 2D numpy array to file.
        The function processes data from a specified simulation result category and axis.
        """

    print("start VTK analysing...")

    # Load configuration from JSON file
    with open("config.json", "r") as file:
        parameters = json.load(file)

    # Extract relevant parameters
    folder = parameters["result_folder"]
    step = read_step_size("field")

    # Load HDF settings from simulation output
    setting = read.ReadHDFSettings(folder + "settings.hdf")
    nx = setting.get_num_cells("x")
    nz = setting.get_num_cells("z")
    dt = setting.get_time_step_size()
    num_of_files = setting.get_time_step_cycles()

    # Extract variable name and axis from input set_data
    name_split = set_data.split("_")
    category = name_split[0] if name_split else ""
    axis = name_split[1] if len(name_split) > 1 else "x"

    print(f"data quatity to analyze: {category}, specification: {axis}")

    # Mapping for known vector fields
    field_map = {"Efield": "E", "Bfield": "B"}

    # Special cases where the variable name includes the axis
    special_cases = {"rhoe", "rhoi", "Je", "Ji"}

    # Determine the actual variable name to look for
    if category in field_map:
        variable_name = field_map[category]
    elif (category+axis) in special_cases:
        variable_name = category + axis
    else:
        variable_name = category

    # Sort and select relevant files for processing
    par_sorted_files = process_folder(folder, "_" + str(variable_name) + "_", extract_number)

    # Read data from VTK files
    pic_data = read.ReadVTKFilesData(folder, par_sorted_files, num_of_files, step, dt, nx, nz, variable_name, axis)

    # Mapping categories to the desired output variable names
    proc_var_dict = {
        "Efield": "Efield",
        "Bfield": "Bfield",
        "rhoe0": "rhoe0",
        "rhoi1": "rhoi1",
        "rhoe2": "rhoe2",
        "Je": "electron_current",
        "Ji": "ion_current",
    }

    print(f"Are data in vector format: {pic_data.is_vector()}")

    # Reconstruct the set_data string based on whether it's a vector field
    if pic_data.is_vector():
        set_data = f"{proc_var_dict[category]}_{axis}"
    else:
        set_data = f"{proc_var_dict[category]}"

    # Save processed 2D data as .npy
    output_file = "script_data/" + set_data + "_data2D.npy"

    np.save(output_file, pic_data.get_2D_data())
    print(f"new file created: {output_file}")

    print("VTK analysing done")


def conserve_analysis(set_data):
    """
        Analyze conserved quantities (like kinetic or electric energy)
        from simulation results and save them as a 1D NumPy array.

        Parameters:
            set_data (str): The key indicating which type of energy to analyze.
                            Supported values: "energy_kin", "energy_ele"
        """
    # Load configuration from JSON
    with open("config.json", "r") as file:
        parameters = json.load(file)

    folder = parameters["result_folder"]

    # Read conserved quantities from file
    energy_data = read.ReadConsData(folder, "ConservedQuantities.txt")

    # Define supported types of conserved energy data
    supported = {
        "energy_kin": energy_data.get_k_energy,
        "energy_ele": energy_data.get_e_energy,
    }

    # Check if the requested data is available
    if set_data in supported:
        new_file_name = f"script_data/{set_data}_data1D.npy"

        # Call the appropriate method and extract the values
        data = supported[set_data]()  # returns tuple (x, y)
        np.save(new_file_name, data[1])  # Save only y-values

        print(f"New file created: {new_file_name}")
        return

    # If unsupported data type is requested
    print(f"Data '{set_data}' not available")
    raise ValueError(f"Data '{set_data}' not available")



