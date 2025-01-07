# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json
from post_process.read_files import ReadFilesData, min_value, max_value
import unit_convert
from post_process import ploting
from distutils.util import strtobool
from pre_process.unit_input import particles_parameters, const_e, const_M_e, T_e, n_e, T_i, const_M_pr, n_i



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
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

    # create object with simulation data
    PIC_data = ReadFilesData(folder, file_for_graph, num_of_files, step, dt, nx, nz)
    print(f" Is vector? {PIC_data.is_vector()}")
    data_x_t = PIC_data.get_2D_data("x")
    data_name = PIC_data.get_data_name()
    axis_time = PIC_data.get_time_data()
    axis_x = PIC_data.get_x_data(Lx)

    # units convert
    electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
    ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)

    axis_x_SI = unit_convert.rescale_list(axis_x, ion.get_ion_skin_depth())
    axis_time_SI = unit_convert.rescale_list(axis_time, 1 / ion.get_plasma_frequency())
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

        val1 = PIC_data.get_point_through_len(x_level, "x")
        descr1 = ploting.PlotDescription(f"data {data_name} for time {axis_time[x_level]}", "length [isd]",
                                     "el. intensity [ ]")

        # ploting.plot_data(axis_x, val1, descr1, False, "plot_E(x)")

        descr11 = ploting.PlotDescription(f"data {data_name} for t = {round(axis_time_SI[x_level], 3)} s", "length [m]",
                                      "el. intensity [ ]")
        descr11.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
        ploting.plot_data(axis_x_SI, val1, descr11, save, "plot_E_SI(x)")

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_time"]["show"])):
        # val2 = get_point_through_time(data_x_t, 4094, PIC_data.is_vector(), "x")
        t_level = parameters["visualization_parameters"]["plot_time"]["length_parameter"]
        save = bool(strtobool(parameters["visualization_parameters"]["plot_time"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_time"]["file_name"]

        val2 = PIC_data.get_point_through_time(t_level, "x")
        print(f"len val2: {len(val2)}")
        print(f"len x_axis: {len(axis_x)}")
        descr2 = ploting.PlotDescription(f"data {data_name} for len {axis_x[4094]}", "time [pf]", "el. intensity [ ]")
        # ploting.plot_data(axis_time, val2, descr2, True, "plot3D_E(t)")

        descr22 = ploting.PlotDescription(f"data {data_name} for x = {round(axis_x_SI[t_level], 1)} m", "time [s]",
                                      "el. intensity [ ]")
        descr22.set_ylim(min_value(data_x_t) * 1.1, max_value(data_x_t) * 1.1)
        ploting.plot_data(axis_time_SI, val2, descr22, save, save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_3D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_3D"]["file_name"]

        descr3D = ploting.PlotDescription(f"time development for {data_name}", "length [m]", "time [s]",
                                      "el. intensity [ ]")
        ploting.plot3D_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D, save, save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_wireframe"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_wireframe"]["file_name"]

        descr3D = ploting.PlotDescription(f"time development for {data_name}", "length [m]", "time [s]",
                                          "el. intensity [ ]")
        ploting.plot3Dwire_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D, save, save_file_name)

    # --------------------------------------------------------------------------------------------------
    if bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["show"])):
        save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
        save_file_name = parameters["visualization_parameters"]["plot_2D"]["file_name"]

        descr3D = ploting.PlotDescription(f"time development for {data_name}", "length [m]", "time [s]",
                                          "el. intensity [ ]")
        ploting.plot3Dplane_data(axis_x_SI, axis_time_SI, PIC_data.get_2D_data(), descr3D, save, save_file_name)

    ploting.plot_all_graphs()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
