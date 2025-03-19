
import numpy as np
import json
import post_process.ploting as ploting
import unit_convert
import pre_process.unit_input as unit
from post_process.vtk_processing import read_step_size
import post_process.read_files as read

# d = np.load("../script_data/Efield_x_data2D.npy")
# a = np.load("../script_data/velocity_1_x_data2D.npy")

# print(d[0][0])
# print(a[0][0])

class ReadNMPFileData:
    def __init__(self, set_data):
        name_split = set_data.split("_")
        category = name_split[0] if name_split else ""
        axis = name_split[1] if len(name_split) > 1 else "x"

        data_name = "script_data/" + set_data + "_data2D.npy"
        print(data_name)
        self.data = np.load(data_name)

    def get_data_x_t(self):
        return self.data

    def get_time_axis(self):

        with open("parameters_hdf5.json", "r") as file:
            parameters = json.load(file)

        # read parameters to variables
        folder = parameters["folder"]
        setting = read.ReadHDFSettings(folder + "settings.hdf")
        dt = setting.get_time_step_size()
        step = read_step_size()
        axis_time = []
        for i in range(0, len(self.data)):
            axis_time.append(i * step * dt)

        return axis_time

    def get_length_axis(self):
        axis_x = []
        for i in range(0, len(self.data[0])):
            axis_x.append(i)

        return axis_x

# def load_numpy_data(set_data):
#     name_split = set_data.split("_")
#     category = name_split[0] if name_split else ""
#     axis = name_split[1] if len(name_split) > 1 else "x"
#
#     data_name = "script_data/" + set_data + "_data2D.npy"
#     print(data_name)
#     d = np.load(data_name)
#     return d

def result_analysis(set_data):

    print(set_data)

    numpy_data = ReadNMPFileData(set_data)
    data = numpy_data.get_data_x_t()

    print(f"len T: {len(data)}")
    print(f"len L:{len(data[0])}")
    print("Numpy data loaded")

    axis_name = {
        "Efield": "Electric field [V/m]",
        "Bfield": "Magnetic field []",
        "rhoe": "Electron density []",
        "rhoi": "Ion density []",
        "Je": "Electron current density []",
        "Ji": "Ion current density []",
    }

    data_si = unit_convert.rescale_list_of_lists(data, unit.c1.e_field_const)

    x_level = -1

    axis_time = numpy_data.get_time_axis()
    print(axis_time)
    print(axis_time[-1]/0.0001)

    axis_x = numpy_data.get_length_axis()
    axis_x_SI = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * 0.5/4096)

    print(f"plasma_frequency: {unit.ion.get_plasma_frequency()}")
    axis_time_SI = unit_convert.rescale_list(axis_time, 1 / unit.ion.get_plasma_frequency())
    axis_time_SIms = unit_convert.rescale_list(axis_time_SI, 1000)

    # debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)
    axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1 / unit.debye_len)
    axis_time_OM = unit_convert.rescale_list(axis_time_SI, unit.electron.get_plasma_frequency())

    print(f"length DB max: {axis_x_DB[-1]}")
    print(f"time OM max: {axis_time_OM[-1]}")

    # graph parameters for processing
    # x_level = -1
    save_file_name = set_data + "_" + "hdf5_len.png"
    val1 = data_si[x_level]

    y_axis_name = set_data.split("_")[0]
        # plot data directly
    descr11 = ploting.PlotDescription(f"Length data {set_data} for t = {round(axis_time_SIms[x_level], 3)} ms",
                                          "Length [m]", axis_name[y_axis_name])
        # descr11.set_ylim(read.min_value(data_x_t) * 0.95, read.max_value(data_x_t) * 1.05)
    ploting.plot_data(axis_x_SI, val1, descr11, False, save_file_name)

        # plot data with FFT
        # if enable_fft:
    descr22 = ploting.PlotDescription(
                f"Frequency Spectrum for {set_data}; t = {round(axis_time_SIms[x_level], 1)} ms",
                "Wavenumber [m-1]",
                "Magnitude")
    fft_file_name = read.add_suffix(save_file_name, "_fft.")
    ploting.plot_fft(axis_x_SI, val1, descr22, False, fft_file_name)

    # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
    # save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_2D"]["file_name"]

    descr3D = ploting.PlotDescription(f"Time development through space for {set_data}", r"$Length~~[\lambda_{D}]$",
                                          r"$Time~~[1/\omega_{pi}]$",
                                          axis_name[y_axis_name])
    descr3D.set_ylim(read.min_value(data), read.max_value(data))
    ploting.plot3Dplane_data(axis_x_DB, axis_time_OM, data, descr3D, False, save_file_name)
