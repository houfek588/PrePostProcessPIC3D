import post_process.read_files as read
from post_process import ploting
import unit_convert
import pre_process.unit_input as unit

def result_analysis():
    path = "../res_data_vth/beam01_drftB/data_hdf5/"
    file = "restart0.hdf"
    number_files = 32

    setting = read.ReadHDFSettings("../res_data_vth/beam01_drftB/data_hdf5/settings.hdf")

    print(setting.get_num_cycles())

    particle_dict = {"electron": "species_0",
                     "ion": "species_1"}
    velocity_dict = {"x": "u",
                     "y": "v",
                     "u": "w"}

    f_file = read.ReadHDFFieldData(path, file, number_files, "E", "x")
    # p_file = read.ReadHDFParticleData(path, file, number_files, particle_dict["electron"], velocity_dict["x"])
    p_file = read.ReadHDFParticleData(path, file, number_files, particle_dict["electron"], "q")

    axis_x = p_file.get_len_data()
    axis_x_dict = {"pic": axis_x,
                   "SI": unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth())
    }

    axis_field_t = f_file.get_time_data()
    axis_t_dict = {"pic": axis_field_t,
                   "SI": unit_convert.rescale_list(axis_field_t, 1 / unit.ion.get_plasma_frequency())
                   }



    if True:

        # graph parameters for processing
        # x_level = parameters["visualization_parameters"]["plot_length"]["time_parameter"]
        # save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_length"]["file_name"]
        # enable_fft = bool(strtobool(parameters["visualization_parameters"]["plot_length"]["enable_fft"]))

        t = -1
        # load data for graph
        velocity = p_file.get_field1D_len(t)

        velocity_SI = unit_convert.rescale_list(velocity, unit.const_c)

        # plot data directly
        descr11 = ploting.PlotDescription(f"Length data for t = {round(axis_t_dict['SI'][t], 3)} s",
                                          "length [m]", "velocity [m/s]")
        # descr11.set_ylim(min_value(data_x_t) * 0.95, max_value(data_x_t) * 1.05)
        ploting.plot_data(axis_x_dict["SI"], velocity, descr11)

        # plot data with FFT
        # if enable_fft:
        #     descr22 = ploting.PlotDescription(
        #         f"Frequency Spectrum for {data_name}; t = {round(axis_time_SIms[x_level], 1)} ms",
        #         "Wavenumber [m-1]",
        #         "Magnitude")
        #     fft_file_name = add_suffix(save_file_name, "_fft.")
        #     ploting.plot_fft(axis_x_SI, val1, descr22, save, save_file_path + fft_file_name)

        # --------------------------------------------------------------------------------------------------
    if True:
        # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
        # save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_2D"]["file_name"]

        Z = f_file.get_2D_data()

        descr3D = ploting.PlotDescription(f"Time development through space for E", "length [db]",
                                              "time [1/Om_pi]",
                                              "E")
        # descr3D.set_ylim(min_value(data_x_t), max_value(data_x_t))
        ploting.plot3Dplane_data(axis_x, axis_field_t, Z, descr3D)

    ploting.plot_all_graphs()