# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
import post_process.read_files as read
import unit_convert
from post_process import ploting
from distutils.util import strtobool
import pre_process.unit_input as unit
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# def get_axis_data(data):
#     axis_x = data.get_len_data()
#     axis_x_dict = {"pic": axis_x,
#                    "SI": unit_convert.rescale_list(axis_x, unit_convert.ion.get_ion_skin_depth())
#                    }
#
#     axis_t = data.get_time_data()
#     axis_t_dict = {"pic": axis_t,
#                    "SI": unit_convert.rescale_list(axis_t, 1 / unit_convert.ion.get_plasma_frequency())
#                    }
#
#     return [axis_x_dict, axis_t_dict]


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


def read_step_size():
    with open("parameters_vtk.json", "r") as file:
        parameters = json.load(file)
    return parameters["sim_parameters"]["step"]


def result_analysis(set_data):
    print("start vtk analysing...")
    with open("parameters_vtk.json", "r") as file:
        parameters = json.load(file)



    # read parameters to variables
    folder = parameters["folder"]

    name_split = set_data.split("_")
    category = name_split[0] if name_split else ""
    axis = name_split[1] if len(name_split) > 1 else "x"

    print(f"{category}, {axis}")

    # Mapping specific categories to their shorthand
    field_map = {"Efield": "E", "Bfield": "B"}

    # Special cases that should be concatenated with axis
    special_cases = {"rhoe", "rhoi", "Je", "Ji"}

    if category in field_map:
        variable_name = field_map[category]
    elif (category+axis) in special_cases:
        variable_name = category + axis
    else:
        variable_name = category

    # variable_name = proc_var

    # file_for_graph = parameters["sim_name"] + "_" + variable_name + "_0." + parameters["data_type"]
    # num_of_files = parameters["sim_parameters"]["num_of_files"]
    step = parameters["sim_parameters"]["step"]

    setting = read.ReadHDFSettings(folder + "settings.hdf")
    nx = setting.get_num_cells("x")
    nz = setting.get_num_cells("z")
    dt = setting.get_time_step_size()
    num_of_files = setting.get_time_step_cycles()

    # axis = parameters["axis"]

    print("files to analyze: " + parameters["sim_name"] + "_" + variable_name + "_xxx." + parameters["data_type"])
    # print(folder, "_" + str(variable_name) + "_")
    par_sorted_files = process_folder(folder, "_" + str(variable_name) + "_", extract_number)


    # create object with simulation data
    PIC_data = read.ReadVTKFilesData(folder, par_sorted_files, num_of_files, step, dt, nx, nz, variable_name, axis)
    # energy_data = read.ReadConsData(folder, "ConservedQuantities.txt")

    # read data from loaded file
    print(f"is vector data? {PIC_data.is_vector()}")
    # if variable_name == "E":
    #     PIC_data.rescale_data(unit.c1.e_field_const)
    #     # data_x_t = unit_convert.rescale_list_of_lists(PIC_data.get_2D_data(),unit.c1.e_field_const)
    #     print(f"data f(x,t) was rescaled by {unit.c1.e_field_const}")

    # data_x_t = PIC_data.get_2D_data()
    # np.save(parameters["output_folder"] + variable_name + "_data2D.npy", PIC_data.data_x_t)
    proc_var_dict = {
        "E": category,
        "B": category,
        "rhoe": "rho",
        "rhoi": "rho",
        "Je": "electron_current",
        "Ji": "ion_current",
    }

    set_data = f"{proc_var_dict[variable_name]}_{axis}"
    new_file_name = "script_data/" + set_data + "_data2D.npy"
    np.save(new_file_name, PIC_data.get_2D_data())
    print(f"new file created on {new_file_name}")

    # if PIC_data.is_vector():
    #     data_name = PIC_data.get_data_name() + "_" + axis
    # else:
    #     data_name = PIC_data.get_data_name()
    # axis_time = PIC_data.get_time_data()
    # axis_x = PIC_data.get_len_data(Lx)

    # axis_name = {
    #     "E": "Electric field [V/m]",
    #     "B": "Magnetic field []",
    #     "rhoe": "Electron density []",
    #     "rhoi": "Ion density []",
    #     "Je": "Electron current density []",
    #     "Ji": "Ion current density []",
    # }

    # units convert
    # electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
    # ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)

    # axis_x_SI = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * Lx/nx)
    # axis_time_SI = unit_convert.rescale_list(axis_time, 1 / unit.ion.get_plasma_frequency())
    # axis_time_SIms = unit_convert.rescale_list(axis_time_SI, 1000)
    #
    # # debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)
    # axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1 / unit.debye_len)
    # axis_time_OM = unit_convert.rescale_list(axis_time_SI, unit.electron.get_plasma_frequency())
    #
    # # print(energy_data.get_e_energy()[1][-1])
    # energy_si = unit_convert.rescale_list(energy_data.get_e_energy()[1], unit.c1.energy_const)
    # print(f"data f(x,t) was rescaled by {unit.c1.energy_const}")
    # # print(energy_si[-1])
    #
    # # print(energy_data.get_e_energy()[1][-1])
    # k_energy_si = unit_convert.rescale_list(energy_data.get_k_energy()[1], unit.c1.energy_const)
    # print(f"data f(x,t) was rescaled by {unit.c1.energy_const}")
    #
    # print("analysis completed")
    # print("graph calculation...")
    #
    # save = bool(strtobool(parameters["save_graphs"]))
    # save_file_path = parameters["output_folder"]



    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    # visualization parameters
    # if bool(strtobool(parameters["visualization_parameters"]["plot_length"]["show"])):
    #
    #     # graph parameters for processing
    #     x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_length"]["file_name"]
    #     enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["enable_fft"]))
    #
    #     # load data for graph
    #     val1 = PIC_data.get_field1D_len(x_level)
    #
    #     # plot data directly
    #     descr11 = ploting.PlotDescription(f"Length data {data_name} for t = {round(axis_time_SIms[x_level], 3)} ms",
    #                                       "Length [m]", axis_name[PIC_data.get_data_name()])
    #     # descr11.set_ylim(read.min_value(data_x_t) * 0.95, read.max_value(data_x_t) * 1.05)
    #     ploting.plot_data(axis_x_SI, val1, descr11, save, save_file_path + save_file_name)
    #
    #     # plot data with FFT
    #     if enable_fft:
    #         descr22 = ploting.PlotDescription(
    #             f"Frequency Spectrum for {data_name}; t = {round(axis_time_SIms[x_level], 1)} ms",
    #             "Wavenumber [m-1]",
    #             "Magnitude")
    #         fft_file_name = read.add_suffix(save_file_name, "_fft.")
    #         ploting.plot_fft(axis_x_SI, val1, descr22, save, save_file_path + fft_file_name)
    #
    # # --------------------------------------------------------------------------------------------------
    # if bool(strtobool(parameters["visualization_parameters"]["plot_time"]["show"])):
    #
    #     # graph parameters for processing
    #     t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_time"]["file_name"]
    #     enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_time"]["enable_fft"]))
    #
    #     # load data for graph
    #     val2 = PIC_data.get_field1D_time(t_level)
    #
    #     # plot data directly
    #     descr22 = ploting.PlotDescription(f"Time data {data_name} for x = {round(axis_x_SI[t_level], 1)} m", "Time [s]",
    #                                       axis_name[PIC_data.get_data_name()])
    #     descr22.set_ylim(read.min_value(data_x_t) * 0.95, read.max_value(data_x_t) * 1.05)
    #     ploting.plot_data(axis_time_SI, val2, descr22, save, save_file_path + save_file_name)
    #
    #     # plot data with FFT
    #     if enable_fft:
    #         descr22 = ploting.PlotDescription(
    #             f"Frequency Spectrum for {data_name} for x = {round(axis_x_SI[t_level], 1)} m",
    #             "Frequency (Hz)",
    #             "Magnitude")
    #         fft_file_name = read.add_suffix(save_file_name, "_fft.")
    #         ploting.plot_fft(axis_time_SI, val2, descr22, save, save_file_path + fft_file_name)
    #
    # # --------------------------------------------------------------------------------------------------
    # if bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["show"])):
    #     # save = bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["save_to_file"]))
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_3D"]["file_name"]
    #
    #     descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "Length [m]", "Time [ms]",
    #                                       axis_name[PIC_data.get_data_name()])
    #     ploting.plot3D_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_path + save_file_name)
    #
    # # --------------------------------------------------------------------------------------------------
    # if bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["show"])):
    #     # save = bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["save_to_file"]))
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_wireframe"]["file_name"]
    #
    #     descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "Length [m]", "Time [s]",
    #                                       axis_name[PIC_data.get_data_name()])
    #     ploting.plot3Dwire_data(axis_x_SI, axis_time_SI, data_x_t, descr3D, save, save_file_path + save_file_name)
    #
    # # --------------------------------------------------------------------------------------------------
    # if bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["show"])):
    #     # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_2D"]["file_name"]
    #
    #     descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "Length [db]",
    #                                       r"$Time~~[1/\omega_{pi}]$",
    #                                       axis_name[PIC_data.get_data_name()])
    #     descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    #     ploting.plot3Dplane_data(axis_x_DB, axis_time_OM, data_x_t, descr3D, save, save_file_path + save_file_name)
    #
    # # --------------------------------------------------------------------------------------------------
    # if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["show"])):
    #     t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
    #     x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
    #     # save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["save_to_file"]))
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_FFT_1D"]["file_name"]
    #
    #     # val = PIC_data.get_point_through_time(t_level, axis)
    #     # x_ax = axis_time_SI
    #
    #     val = PIC_data.get_point_through_len(x_level, axis)
    #     x_ax = axis_x_SI
    #
    #     aa = 50
    #     step = x_ax[aa] - x_ax[aa - 1]
    #     print(f"step: {step}")
    #     print(f"len(x_ax): {len(x_ax)}")
    #
    #     # Perform FFT
    #     fft_result = np.fft.fft(val)
    #     frequencies = np.fft.fftfreq(len(x_ax), d=step)
    #
    #     # Get magnitude spectrum (optional)
    #     magnitude = np.abs(fft_result)
    #
    #     descr22 = ploting.PlotDescription(f"Frequency Spectrum for {data_name}; x = {round(axis_x_SI[t_level], 1)} m",
    #                                       "Frequency (Hz)",
    #                                       "Magnitude")
    #     # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
    #     ploting.plot_data(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2], descr22, save,
    #                       save_file_path + save_file_name)
    #
    # if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["show"])):
    #     # save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["save_to_file"]))
    #     save_file_name = parameters["visualization_parameters"]["plot_FFT_2D"]["file_name"]
    #
    #     # Create a 2D array (e.g., a Gaussian function as example data)
    #     # x = axis_x
    #     # y = axis_time
    #     # x = axis_x_SI
    #     # y = axis_time_SI
    #     x = axis_x_DB
    #     y = axis_time_OM
    #     # x = unit_convert.rescale_list(axis_x, unit.debye_len)
    #     # y = unit_convert.rescale_list(axis_time_SI, 1/unit.electron.get_plasma_frequency())
    #     X, Y = np.meshgrid(x, y)
    #     Z = np.array(data_x_t)  # Example 2D Gaussian
    #
    #     fft_results = read.fft_2d(Z, x, y)
    #
    #     # axis_x_DB, axis_time_OM
    #
    #     if fft_results:
    #         E_fft_shifted, kx_shifted, ky_shifted = fft_results
    #
    #         print("PRINT FFT 2D")
    #
    #         # Plot the results
    #         # plt.figure(figsize=(12, 6))
    #
    #         # plt.subplot(121)
    #         # plt.imshow(np.abs(Z), extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto')
    #         # plt.imshow(np.abs(Z), extent=[min(x), max(x), min(y), max(y)], origin='lower', aspect='auto')
    #         # plt.imshow(Z, extent=[min(x), max(x), min(y), max(y)], origin='lower', aspect='auto', cmap='viridis')
    #         # plt.title(r"$Time~evolution~of~E_x$")
    #         # plt.xlabel("Length [db]")
    #         # plt.ylabel(r"$Time~~[1/\omega_{pe}]$")
    #
    #         descr3D = ploting.PlotDescription(r"$Time~evolution~of~E_x$", r"$Length~~[\lambda_{D}]$",
    #                                           r"$Time~~[1/\omega_{pe}]$",
    #                                           axis_name[PIC_data.get_data_name()])
    #         # descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    #         ploting.plot3Dplane_data(x, y, Z, descr3D, save, save_file_path + save_file_name)
    #
    #         # plt.subplot(122)
    #         # plt.imshow(np.abs(E_fft_shifted),
    #         #            extent=[kx_shifted.min(), kx_shifted.max(), ky_shifted.min(), ky_shifted.max()], origin='lower',
    #         #            aspect='auto')
    #         # plt.imshow(np.abs(E_fft_shifted),
    #         #            extent=[0, max(kx_shifted), 0, max(ky_shifted)], origin='lower',
    #         #            aspect='auto', cmap='viridis')
    #         # plt.title(r"$FFT~2D~result~of~E_x$")
    #         # plt.xlabel("Wavenumber [1/db]")
    #         # plt.ylabel(r"$Frequency~~[\omega_{pe}]$")
    #         # # fig.colorbar(im, ax=ax, label=descr.label_z)
    #         # plt.tight_layout()
    #         # plt.show()
    #
    #         descr3D = ploting.PlotDescription(r"$FFT~2D~result~of~E_x$", r"$Wavenumber~~[1/\lambda_{D}]$",
    #                                           r"$Frequency~~[\omega_{pe}]$",
    #                                           "Magnitude")
    #         # descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    #         ploting.plot3Dplane_data(kx_shifted, ky_shifted, np.abs(E_fft_shifted), descr3D, save,
    #         save_file_path + "fft_" + save_file_name)
    #         # :len(frequencies) // 2
    #         # ploting.plot3Dplane_data(kx_shifted[:len(kx_shifted) // 2], ky_shifted[:len(ky_shifted) // 2],
    #         #                          np.abs(E_fft_shifted[:len(kx_shifted) // 2, :len(ky_shifted) // 2]), descr3D, save,
    #         #                          save_file_path + "fft_" + save_file_name)
    #
    #
    #         # ploting.plot_data(kx_shifted, ky_shifted, descr3D)
    #
    #         # If you need the phase:
    #         # phase = np.angle(E_fft_shifted)
    #         # plt.figure()
    #         # # plt.imshow(phase, extent=[kx_shifted.min(), kx_shifted.max(), ky_shifted.min(), ky_shifted.max()],
    #         # #            origin='lower', aspect='auto')
    #         # plt.imshow(phase, extent=[min(kx_shifted), max(kx_shifted), min(ky_shifted), max(ky_shifted)],
    #         #            origin='lower', aspect='auto')
    #         # plt.title("FFT (Phase)")
    #         # plt.xlabel("kx")
    #         # plt.ylabel("ky")
    #         # plt.show()
    #
    #     # Perform 2D FFT
    #     # fft_result = np.fft.fft2(Z)
    #     # fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency to the center
    #     # magnitude = np.abs(fft_shifted)  # Magnitude of the FFT result
    #     #
    #     # # fr = np.fft.
    #     #
    #     # print("FFT 2D")
    #     # print(f"type: {type(fft_shifted)}, length: {len(fft_shifted)}")
    #     # print(f"type: {type(fft_shifted[0])}, length: {len(fft_shifted[0])}")
    #     # # print(f"type: {type(fft_shifted[0][0])}, length: {len(fft_shifted[0][0])}")
    #     # # norm_om = unit_convert.rescale_list(axis_time_OM, 1/unit.electron.get_plasma_frequency())
    #     #
    #     # descr3D = ploting.PlotDescription(f"FFT 2D Result {data_name}", "Length [db]",
    #     #                                   r"$Frequency~~[\omega_{pi}]$", "Magnitude")
    #     # descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    #     # ploting.plot3Dplane_data(axis_x_DB, axis_time_OM, magnitude, descr3D, save, save_file_path + save_file_name)
    #
    #
    #
    #
    # if False:
    #     # graph parameters for processing
    #     save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_length"]["file_name"]
    #     enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["enable_fft"]))
    #
    #     # load data for graph
    #
    #
    #     description = ploting.PlotDescription(f"Time development el. field energy E", r"$Time~~[1/\omega_{pi}]$",
    #                                   energy_data.get_e_energy()[0] + " [J]")
    #
    #     print(f"cycles: {axis_x[-1]} = {energy_data.get_cycles()[1][-1]}")
    #     ploting.plot_data(axis_time_OM, energy_si, description,
    #                       save,  save_file_path + "field" + save_file_name)
    #
    #     description = ploting.PlotDescription(f"Time development kinetic energy E", r"$Time~~[1/\omega_{pi}]$",
    #                                           energy_data.get_k_energy()[0] + " [J]")
    #     ploting.plot_data(axis_time_OM, k_energy_si, description,
    #                       save, save_file_path + "kin" + save_file_name)
    #     # ploting.plot_all_graphs()
    #
    #
    #
    #
    # print("graph calculation done")
    #
    # np.save(parameters["output_folder"] + variable_name + "_data2D.npy", PIC_data.data_x_t)
    #
    # print("plot...")
    # # ploting.plot_all_graphs()

    print("vtk analysing done")


def conserve_analysis(set_data):

    with open("parameters_vtk.json", "r") as file:
        parameters = json.load(file)
    folder = parameters["folder"]

    energy_data = read.ReadConsData(folder, "ConservedQuantities.txt")
    available = ["energy_kin", "energy_ele"]
    for a in available:
        if set_data == a:
            # set_data = f"{proc_var_dict[variable_name]}_{axis}"
            new_file_name = "script_data/" + set_data + "_data1D.npy"
            # print(energy_data.get_e_energy())
            if set_data == "energy_kin":
                np.save(new_file_name, energy_data.get_k_energy()[1])
            elif set_data == "energy_ele":
                np.save(new_file_name, energy_data.get_e_energy()[1])
            else:
                raise Exception(f"not implemented analysis for data: {set_data}")
            print(f"new file created on {new_file_name}")
            break
