# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
from post_process.read_files import ReadVTKFilesData, min_value, max_value, add_suffix
import unit_convert
from post_process import ploting
from distutils.util import strtobool
from pre_process.unit_input import (particles_parameters, const_e, const_M_e, T_e, n_e, T_i, const_M_pr, n_i,
                                    get_debey_length, const_eps_0, const_K_b)

import numpy as np
import matplotlib.pyplot as plt

def result_analysis(proc_var):
    print("start analysing...")
    with open("parameters.json", "r") as file:
        parameters = json.load(file)

    # read parameters to variables
    folder = parameters["folder"]
    # variable_name = parameters["data_to_analyze"]
    variable_name = proc_var
    file_for_graph = parameters["sim_name"] + "_" + variable_name + "_0." + parameters["data_type"]
    # file_for_graph = parameters["datafile"]
    num_of_files = parameters["sim_parameters"]["num_of_files"]
    step = parameters["sim_parameters"]["step"]
    dt = parameters["sim_parameters"]["dt"]
    Lx = parameters["sim_parameters"]["Lx"]
    nx = parameters["sim_parameters"]["nx"]
    nz = parameters["sim_parameters"]["nz"]
    axis = parameters["axis"]

    print("files to analyze: " + parameters["sim_name"] + "_" + variable_name + "_xxx." + parameters["data_type"])

    # create object with simulation data
    PIC_data = ReadVTKFilesData(folder, file_for_graph, num_of_files, step, dt, nx, nz)
    print(f" Is vector? {PIC_data.is_vector()}")
    data_x_t = PIC_data.get_2D_data(axis)
    print(data_x_t[0])
    if PIC_data.is_vector():
        data_name = PIC_data.get_data_name() + "_" + axis
    else:
        data_name = PIC_data.get_data_name()
    axis_time = PIC_data.get_time_data()
    axis_x = PIC_data.get_x_data(Lx)

    axis_name = {
        "E": "Electric field []",
        "B": "Magnetic field []",
        "rhoe": "Electron density []",
        "rhoi": "Ion density []",
        "Je": "Electron current density []",
        "Ji": "Ion current density []",
    }

    # units convert
    electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
    ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)

    axis_x_SI = unit_convert.rescale_list(axis_x, ion.get_ion_skin_depth())
    axis_time_SI = unit_convert.rescale_list(axis_time, 1 / ion.get_plasma_frequency())
    axis_time_SIms = unit_convert.rescale_list(axis_time_SI, 1000)

    debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)
    axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1 / debye_len)
    axis_time_OM = unit_convert.rescale_list(axis_time_SI, electron.get_plasma_frequency())
    # print(f"time {4163 / ion.get_plasma_frequency()}")
    # print(f"time step: {dt} -> {(1 / ion.get_plasma_frequency()) * dt} s")
    # aa = 50
    # print(f"data available for every: {axis_time_SI[aa] - axis_time_SI[aa - 1]} s")

    print("analysis completed")
    print("graph calculation...")

    save = bool(strtobool(parameters["save_graphs"]))
    save_file_path = parameters["output_folder"]

    # --------------------------------------------------------------------------------------------------
    # visualization parameters
    if bool(strtobool(parameters["visualization_parameters"]["plot_length"]["show"])):

        x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_length"]["file_name"]
        enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["enable_fft"]))

        val1 = PIC_data.get_point_through_len(x_level, axis)
        # descr1 = ploting.PlotDescription(f"data {data_name} for time {axis_time[x_level]}", "length [isd]",
        #                              axis_name[PIC_data.get_data_name()])
        # ploting.plot_data(axis_x, val1, descr1, False, "plot_E(x)")

        descr11 = ploting.PlotDescription(f"Length data {data_name} for t = {round(axis_time_SIms[x_level], 3)} ms",
                                          "length [m]", axis_name[PIC_data.get_data_name()])
        descr11.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
        ploting.plot_data(axis_x_SI, val1, descr11, save, save_file_path + save_file_name)

        if enable_fft:
            aa = 50
            step = axis_x_SI[aa] - axis_x_SI[aa - 1]
            # print(f"step: {step}")
            # print(f"len(x_ax): {len(val2)}")

            # Perform FFT
            fft_result = np.fft.fft(val1)
            frequencies = np.fft.fftfreq(len(val1), d=step)

            # Get magnitude spectrum (optional)
            magnitude = np.abs(fft_result)

            descr22 = ploting.PlotDescription(
                f"Frequency Spectrum for {data_name}; t = {round(axis_time_SIms[x_level], 1)} ms",
                "Wavenumber [m-1]",
                "Magnitude")
            # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
            fft_file_name = add_suffix(save_file_name, "_fft.")

            ploting.plot_data(frequencies[1:len(frequencies) // 2], magnitude[1:len(magnitude) // 2], descr22, save,
                              save_file_path + fft_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_time"]["show"])):
        # val2 = get_point_through_time(data_x_t, 4094, PIC_data.is_vector(), "x")
        t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_time"]["save_to_file"]))
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_time"]["file_name"]
        enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_time"]["enable_fft"]))

        val2 = PIC_data.get_point_through_time(t_level, axis)
        print(f"len val2: {len(val2)}")
        print(f"len x_axis: {len(axis_x)}")

        print(f"min_value(data_x_t): {min_value(data_x_t)}")
        print(f"max_value(data_x_t): {max_value(data_x_t)}")

        # descr2 = ploting.PlotDescription(f"data {data_name} for len {axis_x[4094]}", "Time [pf]", axis_name[PIC_data.get_data_name()])
        # ploting.plot_data(axis_time, val2, descr2, True, "plot3D_E(t)")

        descr22 = ploting.PlotDescription(f"Time data {data_name} for x = {round(axis_x_SI[t_level], 1)} m", "Time [s]",
                                          axis_name[PIC_data.get_data_name()])
        descr22.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
        ploting.plot_data(axis_time_SI, val2, descr22, save, save_file_path + save_file_name)

        if enable_fft:
            aa = 50
            step = axis_time_SI[aa] - axis_time_SI[aa - 1]
            # print(f"step: {step}")
            # print(f"len(x_ax): {len(val2)}")

            # Perform FFT
            fft_result = np.fft.fft(val2)
            frequencies = np.fft.fftfreq(len(val2), d=step)

            # Get magnitude spectrum (optional)
            magnitude = np.abs(fft_result)

            descr22 = ploting.PlotDescription(
                f"Frequency Spectrum for {data_name} for x = {round(axis_x_SI[t_level], 1)} m",
                "Frequency (Hz)",
                "Magnitude")
            # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
            fft_file_name = add_suffix(save_file_name, "_fft.")
            ploting.plot_data(frequencies[1:len(frequencies) // 2], magnitude[1:len(magnitude) // 2], descr22, save,
                              save_file_path + fft_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["show"])):
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["save_to_file"]))
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_3D"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [m]", "time [ms]",
                                          axis_name[PIC_data.get_data_name()])
        ploting.plot3D_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_path + save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["show"])):
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["save_to_file"]))
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_wireframe"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [m]", "time [s]",
                                          axis_name[PIC_data.get_data_name()])
        ploting.plot3Dwire_data(axis_x_SI, axis_time_SI, data_x_t, descr3D, save, save_file_path + save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["show"])):
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_2D"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [db]",
                                          "time [1/Om_pi]",
                                          axis_name[PIC_data.get_data_name()])
        descr3D.set_ylim(min_value(data_x_t), max_value(data_x_t))
        ploting.plot3Dplane_data(axis_x_DB, axis_time_OM, data_x_t, descr3D, save, save_file_path + save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["show"])):
        t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
        x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["save_to_file"]))
        save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_FFT_1D"]["file_name"]

        # val = PIC_data.get_point_through_time(t_level, axis)
        # x_ax = axis_time_SI

        val = PIC_data.get_point_through_len(x_level, axis)
        x_ax = axis_x_SI

        aa = 50
        step = x_ax[aa] - x_ax[aa - 1]
        print(f"step: {step}")
        print(f"len(x_ax): {len(x_ax)}")

        # Perform FFT
        fft_result = np.fft.fft(val)
        frequencies = np.fft.fftfreq(len(x_ax), d=step)

        # Get magnitude spectrum (optional)
        magnitude = np.abs(fft_result)

        descr22 = ploting.PlotDescription(f"Frequency Spectrum for {data_name}; x = {round(axis_x_SI[t_level], 1)} m",
                                          "Frequency (Hz)",
                                          "Magnitude")
        # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
        ploting.plot_data(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2], descr22, save,
                          save_file_path + save_file_name)

    if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["show"])):
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_FFT_2D"]["file_name"]

        # descr3D = ploting.PlotDescription(f"time development for {data_name}", "length [m]", "time [ms]",
        #                               "el. field [ ]")
        # ploting.plot3D_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_name)

        # # length fft
        # aa = 50
        # step = axis_x_SI[aa] - axis_x_SI[aa - 1]
        # fft_result = np.fft.fft(val1)
        # frequencies = np.fft.fftfreq(len(val1), d=step)
        #
        # # time FFT
        # aa = 50
        # step = axis_time_SI[aa] - axis_time_SI[aa - 1]
        # fft_result = np.fft.fft(val2)
        # frequencies = np.fft.fftfreq(len(val2), d=step)



        # Create a 2D array (e.g., a Gaussian function as example data)
        x = axis_x_SI
        y = axis_time_SI
        X, Y = np.meshgrid(x, y)
        Z = np.array(data_x_t)  # Example 2D Gaussian

        # Perform 2D FFT
        fft_result = np.fft.fft2(Z)
        fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency to the center
        magnitude = np.abs(fft_shifted)  # Magnitude of the FFT result

        # fr = np.fft.

        print("FFT 2D")
        print(f"type: {type(fft_shifted)}, length: {len(fft_shifted)}")
        print(f"type: {type(fft_shifted[0])}, length: {len(fft_shifted[0])}")
        # print(f"type: {type(fft_shifted[0][0])}, length: {len(fft_shifted[0][0])}")


        descr3D = ploting.PlotDescription(f"FFT 2D Result {data_name}", "Wavenumber [m-1]",
                                          "Frequency (Hz)", "Magnitude")
        descr3D.set_ylim(min_value(data_x_t), max_value(data_x_t))
        ploting.plot3Dplane_data(axis_x_DB, axis_time_OM, (magnitude), descr3D, save, save_file_path + save_file_name)

        # # plt.subplot(1, 2, 2)
        # plt.title("FFT Result (Frequency Domain)")
        # # plt.imshow(np.log1p(magnitude), extent=(-10, 10, -10, 10), cmap='magma')
        # plt.imshow((magnitude), extent=(-10, 10, -10, 10), cmap='viridis')
        # plt.colorbar(label="Log Magnitude")
        # plt.show()



    print("graph calculation done")

    # np.save(parameters["output_folder"] + variable_name + "_data2D.npy", PIC_data.data_x_t)

    print("plot...")
    ploting.plot_all_graphs()

    print("script done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # vars = ["E", "B", "rhoe0", "rhoe2", "rhoi1", "rhoi3"]
    # vars = ["rhoe2", "rhoi3"]
    vars = ["E"]

    for v in vars:
        # result_analysis(v)
        try:
            result_analysis(v)
        except:
            print(f"file not found for {v} variable")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
