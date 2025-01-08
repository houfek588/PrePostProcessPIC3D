# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
from post_process.read_files import ReadFilesData, min_value, max_value
import unit_convert
from post_process import ploting
from distutils.util import strtobool
from pre_process.unit_input import particles_parameters, const_e, const_M_e, T_e, n_e, T_i, const_M_pr, n_i

import numpy as np
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("start analysing...")
    with open("parameters.json", "r") as file:
        parameters = json.load(file)

    # read parameters to variables
    folder = parameters["folder"]
    file_for_graph = parameters["datafile"]
    num_of_files = parameters["sim_parameters"]["num_of_files"]
    step = parameters["sim_parameters"]["step"]
    dt = parameters["sim_parameters"]["dt"]
    Lx = parameters["sim_parameters"]["Lx"]
    nx = parameters["sim_parameters"]["nx"]
    nz = parameters["sim_parameters"]["nz"]
    axis = parameters["axis"]

    # create object with simulation data
    PIC_data = ReadFilesData(folder, file_for_graph, num_of_files, step, dt, nx, nz)
    print(f" Is vector? {PIC_data.is_vector()}")
    data_x_t = PIC_data.get_2D_data(axis)
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
    # print(f"time {4163 / ion.get_plasma_frequency()}")
    # print(f"time step: {dt} -> {(1 / ion.get_plasma_frequency()) * dt} s")
    # aa = 50
    # print(f"data available for every: {axis_time_SI[aa] - axis_time_SI[aa - 1]} s")


    # --------------------------------------------------------------------------------------------------
    # visualization parameters
    if bool(strtobool(parameters["visualization_parameters"]["plot_length"]["show"])):

        x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
        save = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_length"]["file_name"]
        enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["enable_fft"]))

        val1 = PIC_data.get_point_through_len(x_level, axis)
        descr1 = ploting.PlotDescription(f"data {data_name} for time {axis_time[x_level]}", "length [isd]",
                                     axis_name[PIC_data.get_data_name()])

        # ploting.plot_data(axis_x, val1, descr1, False, "plot_E(x)")

        descr11 = ploting.PlotDescription(f"Length data {data_name} for t = {round(axis_time_SIms[x_level], 3)} ms",
                                          "length [m]", "el. field [ ]")
        descr11.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
        # descr11.set_ylim(PIC_data.get_global_min() * 1.1, PIC_data.get_global_max() * 1.1)
        ploting.plot_data(axis_x_SI, val1, descr11, save, "plot_E_SI(x)")

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

            descr22 = ploting.PlotDescription(f"Frequency Spectrum for {data_name}; t = {round(axis_time_SIms[x_level], 1)} ms",
                                          "Wavenumber [m-1]",
                                          "Magnitude")
            # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
            ploting.plot_data(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2], descr22, save,
                          save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_time"]["show"])):
        # val2 = get_point_through_time(data_x_t, 4094, PIC_data.is_vector(), "x")
        t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
        save = bool(strtobool(parameters["visualization_parameters"]["plot_time"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_time"]["file_name"]
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
        ploting.plot_data(axis_time_SI, val2, descr22, save, save_file_name)

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

            descr22 = ploting.PlotDescription(f"Frequency Spectrum for {data_name} for x = {round(axis_x_SI[t_level], 1)} m",
                                          "Frequency (Hz)",
                                          "Magnitude")
            # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
            ploting.plot_data(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2], descr22, save,
                          save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_3D"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [m]", "time [ms]",
                                      axis_name[PIC_data.get_data_name()])
        ploting.plot3D_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_name)


    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_wireframe"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [m]", "time [s]",
                                          axis_name[PIC_data.get_data_name()])
        ploting.plot3Dwire_data(axis_x_SI, axis_time_SI, data_x_t, descr3D, save, save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_2D"]["file_name"]

        descr3D = ploting.PlotDescription(f"Time development through space for {data_name}", "length [m]", "time [ms]",
                                          axis_name[PIC_data.get_data_name()])
        descr3D.set_ylim(min_value(data_x_t),max_value(data_x_t))
        ploting.plot3Dplane_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["show"])):
        t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
        x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
        save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_1D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_FFT_1D"]["file_name"]

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



        descr22 = ploting.PlotDescription(f"Frequency Spectrum for {data_name}; x = {round(axis_x_SI[t_level], 1)} m", "Frequency (Hz)",
                                          "Magnitude")
        # descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
        ploting.plot_data(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2], descr22, save, save_file_name)


    if bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_FFT_2D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_FFT_2D"]["file_name"]

        # descr3D = ploting.PlotDescription(f"time development for {data_name}", "length [m]", "time [ms]",
        #                               "el. field [ ]")
        # ploting.plot3D_data(axis_x_SI, axis_time_SIms, data_x_t, descr3D, save, save_file_name)

        # Create a 2D array (e.g., a Gaussian function as example data)
        x = axis_x_SI
        y = axis_time_SIms
        X, Y = np.meshgrid(x, y)
        Z = np.array(data_x_t)  # Example 2D Gaussian

        # Perform 2D FFT
        fft_result = np.fft.fft2(Z)
        fft_shifted = np.fft.fftshift(fft_result)  # Shift zero frequency to the center
        magnitude = np.abs(fft_shifted)  # Magnitude of the FFT result

        # Plot the original data
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Data (Spatial Domain)")
        plt.imshow(Z, aspect='auto', extent=(axis_x_SI[0], axis_x_SI[-1], axis_time_SIms[0], axis_time_SIms[-1]), cmap='viridis')
        plt.colorbar(label="Amplitude")

        # Plot the FFT result (Frequency Domain)
        plt.subplot(1, 2, 2)
        plt.title("FFT Result (Frequency Domain)")
        # plt.imshow(np.log1p(magnitude), extent=(-10, 10, -10, 10), cmap='magma')
        plt.imshow((magnitude), extent=(-10, 10, -10, 10), cmap='magma')
        plt.colorbar(label="Log Magnitude")
        plt.show()


    print("analysis completed")
    print("graph printing...")
    ploting.plot_all_graphs()

    print("script done")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
