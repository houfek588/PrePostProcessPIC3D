import json
# from bidict import bidict
import post_process.read_files as read
from post_process import ploting
import unit_convert
import pre_process.unit_input as unit
from distutils.util import strtobool
import numpy as np

def result_analysis(set_data):
    """
        Analyze particle data from HDF5 files and save the result as a 2D NumPy array.

        Parameters:
            set_data (str): A string describing the quantity, species, and axis.
                            Format: 'category_numAxis' (e.g., 'velocity_1x')
        """
    print(f"start HDF5 analysis for: {set_data}")
    with open("config.json", "r") as file:
        parameters = json.load(file)

    # Load configuration settings from config.json
    with open("config.json", "r") as file:
        parameters = json.load(file)

    # Extract relevant parameters
    folder = parameters["result_folder"]
    number_files = parameters["NumberHdfFiles"]
    hdf_file = "proc0.hdf"  # Starting HDF5 file name

    # Read simulation settings (e.g., number of particle species)
    setting = read.ReadHDFSettings(folder + "settings.hdf")
    ns = setting.get_num_species()

    # Create a dictionary mapping category/species/index to file symbols
    particle_dict = create_particle_dict(ns)

    # Parse the input string (e.g., 'velocity_1x') into components
    try:
        category, num, axis = set_data.split("_")  # Example: "velocity_1x" -> "velocity", "1", "x"
    except ValueError:
        raise ValueError(f"Invalid format for set_data: '{set_data}'. Expected format 'category_numAxis'.")

    print(f"Category: {category}, Num: {num}, Axis: {axis}")

    # Look up the corresponding species and symbol for the data
    try:
        species, symbol = particle_dict[category][num][axis]
    except KeyError:
        raise KeyError(f"No matching species/symbol found for: {set_data}")

    print(f"Extracting data for species: {species}, symbol: {symbol}")

    # Read the particle data from HDF5 files
    p_file = read.ReadHDFParticleData(folder, hdf_file, number_files, species, symbol)

    # Save the extracted 2D data (time vs. x-axis) to a NumPy file
    new_file_name = f"script_data/{set_data}_data2D.npy"
    np.save(new_file_name, p_file.data_x_t)

    print(f"new file created: {new_file_name}")




def create_particle_dict(ns):
    particle_dict = {"velocity": {}, "position": {}}

    for i in range(1, ns + 1):
        species = f"species_{i-1}"  # Adjust for zero-based index
        particle_dict["velocity"][str(i)] = {"x": [species, "u"], "y": [species, "v"], "z": [species, "w"]}
        particle_dict["position"][str(i)] = {"x": [species, "x"], "y": [species, "y"], "z": [species, "z"]}

    return particle_dict

def particle_graphs(data: read.ReadHDFParticleData, par, sett):
    axis_x_dict = get_axis_data(data, sett)[0]
    axis_t_dict = get_axis_data(data, sett)[1]
    units = "SI"
    # units = "pic"
    data_name = data.get_data_name()

    save = bool(strtobool(par["save_graphs"]))
    save_file_path = par["output_folder"]

    if True:

        # graph parameters for processing
        t = par["visualization_parameters"]["plot_length"]["time_parameter"]
        save_file_name = (par["particle"]["axis"] + "_" + par["particle"]["type"] + "_" +
                          par["visualization_parameters"]["plot_length"]["file_name"])
        enable_fft = bool(strtobool(par["visualization_parameters"]["plot_length"]["enable_fft"]))
        enable_histogram = True

        # t = -1
        # load data for graph
        velocity = data.get_particle1D_len(t)
        velocity_si = unit_convert.rescale_list(velocity, unit.const_c)

        # plot data directly
        description = ploting.PlotDescription(f"Length {data_name} data for t({round(axis_t_dict[units][t], 3)})",
                                              "length [m]", "velocity [m/s]")
        # descr11.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
        ploting.plot_data(axis_x_dict[units], velocity_si, description, save, save_file_path + save_file_name)

        if enable_histogram:
            save_file_name = "hist_" + save_file_name
            description_hist = ploting.PlotDescription(
                f"Histogram for {data_name}; t = {round(axis_t_dict[units][t], 3)} s",
                f"{par['particle']['axis']} [m/s]",
                "Magnitude")
            ploting.plot_histogram(velocity, 4096, description_hist, save, save_file_path + save_file_name)

        # plot data with FFT
        if enable_fft:
            save_file_name = "fft_" + save_file_name
            description_fft = ploting.PlotDescription(
                f"Frequency Spectrum for {data_name}; t = {round(axis_t_dict[units][t], 3)} s",
                "Wavenumber [m-1]",
                "Magnitude")
            # fft_file_name = add_suffix(save_file_name, "_fft.")
            ploting.plot_fft(axis_x_dict["SI"], velocity_si, description_fft)


def field_graphs(data: read.ReadHDFFieldData, par):
    axis_x_dict = get_axis_data(data)[0]
    axis_t_dict = get_axis_data(data)[1]
    units = "SI"

    save = bool(strtobool(par["save_graphs"]))
    save_file_path = par["output_folder"]

    if True:
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
        save_file_name = par["field"]["variable"] + "_" + par["visualization_parameters"]["plot_2D"]["file_name"]

        data_z = data.get_2D_data()

        description3d = ploting.PlotDescription(f"Time development through space for {data.get_data_name()}",
                                                "length [db]", "time [1/Om_pi]", "E")
        # descr3D.set_ylim(min_value(data_x_t), max_value(data_x_t))
        ploting.plot3Dplane_data(axis_x_dict[units], axis_t_dict[units], data_z, description3d, save,
                                 save_file_path + save_file_name)


def get_axis_data(data, setting):
    axis_x = data.get_len_data()
    axis_t = data.get_time_data()

    axis_x_si = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * setting.get_box_size()/setting.get_num_cells())
    axis_time_si = unit_convert.rescale_list(axis_t, 1 / unit.ion.get_plasma_frequency())

    axis_x_dict = {"pic": axis_x,
                   "SI": unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * setting.get_box_size()/setting.get_num_cells()),
                   "db": unit_convert.rescale_list(axis_x_si, 1 / unit.debye_len)
                   }


    axis_t_dict = {"pic": axis_t,
                   "SI": unit_convert.rescale_list(axis_t, 1 / unit.ion.get_plasma_frequency()),
                   "om": unit_convert.rescale_list(axis_time_si, unit.electron.get_plasma_frequency())
                   }

    return [axis_x_dict, axis_t_dict]


def compare_result():
    print("start hdf5 analysing...")
    with open("parameters_hdf5.json", "r") as file:
        parameters = json.load(file)

    # read parameters to variables
    folder = parameters["folder"]
    file = parameters["file"]
    number_files = parameters["num_of_files"]

    particle_par = parameters["particle"]
    setting = read.ReadHDFSettings(folder + "settings.hdf")

    particle_dict = {"electron": "species_0",
                     "ion": "species_1",
                     "electron1": "species_2"}

    el0_file = read.ReadHDFParticleData(folder, file, number_files, particle_dict["electron"], particle_par["axis"])
    el1_file = read.ReadHDFParticleData(folder, file, number_files, particle_dict["electron1"], particle_par["axis"])
    ion_file = read.ReadHDFParticleData(folder, file, number_files, particle_dict["ion"], particle_par["axis"])


    # graph parameters for processing
    # t = par["visualization_parameters"]["plot_length"]["time_parameter"]
    # save_file_name = (par["particle"]["axis"] + "_" + par["particle"]["type"] + "_" +
    #                   par["visualization_parameters"]["plot_length"]["file_name"])
    save = bool(strtobool(parameters["save_graphs"]))
    save_file_path = parameters["output_folder"]
    save_file_name = "electron_comparation.png"
    # enable_fft = bool(strtobool(par["visualization_parameters"]["plot_length"]["enable_fft"]))
    # enable_histogram = True

    # t = -1
    # load data for graph
    velocity_el0 = el0_file.get_particle1D_len(0)
    velocity_el1 = el1_file.get_particle1D_len(0)
    velocity_ion0 = ion_file.get_particle1D_len(0)
    # velocity_si = unit_convert.rescale_list(velocity, unit.const_c)

    # # plot data directly
    # description = ploting.PlotDescription(f"Length {data_name} data for t = {round(axis_t_dict[units][t], 3)} s",
    #                                       "length [m]", "velocity [m/s]")
    # # descr11.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
    # ploting.plot_data(axis_x_dict[units], velocity_si, description, save, save_file_path + save_file_name)

    # if enable_histogram:
    save_file_name = "hist_" + save_file_name
    description_hist = ploting.PlotDescription(
            f"Histogram origin particles velocities; t(0)",
            f"{particle_par['axis']} [c]",
            f"f({particle_par['axis']})")
    description_hist.multidata_labels(["electron_beam", "electron", "ions"])
    ploting.plot_histogram([velocity_el1, velocity_el0, velocity_ion0], 4096, description_hist, save, save_file_path + save_file_name)

    # ploting.plot_all_graphs()