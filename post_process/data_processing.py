
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
g_keys = ["Efield", "Bfield", "rhoe", "rhoi", "Je", "Ji", "velocity", "velocity_1", "velocity_2", "velocity_3",
          "energy", "energy_kin", "energy_ele"]
g_names = ["Electric field ", "Magnetic field", "Electron density", "Ion density", "Electron current density",
           "Ion current density", "Velocity", "Electron velocity", "Ion velocity", "El. beam velocity", "Energy", "Kin. energy",
           "El. energy"]
g_pic_units = ["om_i ", "??", "4pi", "4pi", "??", "??", "c", "c", "c", "c", "??", "??", "??"]
g_si_units = ["V/m ", "T", "1", "1", "A/m", "A/m", "m/s", "m/s", "m/s", "m/s", "J", "J", "J"]

g_axis_name = {key: g_names[i] for i, key in enumerate(g_keys)}
g_axis_si_units = {key: g_si_units[i] for i, key in enumerate(g_keys)}
g_axis_pic_units = {key: g_pic_units[i] for i, key in enumerate(g_keys)}
# g_axis_name = {
#     "Efield": "Electric field ",
#     "Bfield": "Magnetic field []",
#     "rhoe": "Electron density []",
#     "rhoi": "Ion density []",
#     "Je": "Electron current density []",
#     "Ji": "Ion current density []",
#     "velocity": "Velocity"
# }
#
# g_axis_units = {
#     "Efield": "Electric field ",
#     "Bfield": "Magnetic field []",
#     "rhoe": "Electron density []",
#     "rhoi": "Ion density []",
#     "Je": "Electron current density []",
#     "Ji": "Ion current density []",
#     "velocity": "Velocity"
# }

class ReadNMPFileData:
    def __init__(self, set_data):
        name_split = set_data.split("_")
        category = name_split[0] if name_split else ""
        axis = name_split[1] if len(name_split) > 1 else "x"

        try:
            data_name = "script_data/" + set_data + "_data2D.npy"
            self.data = np.load(data_name)
            self.data_dim = 2
        except:
            data_name = "script_data/" + set_data + "_data1D.npy"
            self.data = np.load(data_name)
            self.data_dim = 1
        # print(f"loaded data: {data_name}")

    def get_data_dimension(self):
        return self.data_dim

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
        # print(f"data: {self.data}")
        for i in range(0, len(self.data)):
            axis_time.append(i * step * dt)
        # axis_time = []
        # for i in range(0, len(self.data)):
        #     axis_time.append(i)

        return axis_time

    def get_length_axis(self):
        if self.data_dim == 1:
            return "this fuction is not available for 1D data"

        axis_x = []
        for i in range(0, len(self.data[0])):
            axis_x.append(i)
        return axis_x


def single_result_analysis(set_data):

    print(f"Single data for analysis: {set_data}")

    numpy_data = ReadNMPFileData(set_data)

    if numpy_data.get_data_dimension() == 1:
        graph_energy(numpy_data, set_data, "val", "si")
    else:
        graph_for_pos_value(numpy_data, set_data, 0.5, "val_ms", "si")
        graph2d_fft_in_time(numpy_data, set_data)
    # function calls
    # graph_for_time_value(numpy_data, set_data, 0, "hist", "pic")
    # graph_for_pos_value(numpy_data, set_data, 1, "val_ms", "si")
    # graph_for_time_value(numpy_data, set_data, 1, True)
    # graph2d(numpy_data, set_data)
    # graph2d_fft_in_time(numpy_data, set_data)
    # graph_fft2d(numpy_data, set_data)

    #


def multi_result_analysis(set_data: list):
    print(f"Multiple data for analysis: {set_data}")

    numpy_data = [ReadNMPFileData(name) for name in set_data]
    # for i
    # numpy_data = ReadNMPFileData(set_data)

    # graph_multi_time_value(numpy_data[0], set_data[0], 1, "hist", "pic")
    graph_multi_time_value(numpy_data, set_data, 0, "hist", "pic")
    graph_multi_time_value(numpy_data, set_data, 1, "hist", "pic")
    # function calls
    # graph_for_time_value(numpy_data, set_data, 0, "hist", "pic")
    # graph_for_time_value(numpy_data, set_data, 1, True)
    # graph2d(numpy_data, set_data)
    # graph2d_fft_in_time(numpy_data, set_data)
    # graph_fft2d(numpy_data, set_data)

def graph_for_time_value(numpy_data, set_data, T, res_type = "val", units="si"):
    """Generates a graph for a given dataset at a specific time with optional FFT analysis.

        Args:
            numpy_data: Object containing the dataset.
            set_data: String representing the dataset name.
            T: Time index factor (1 means last entry).
            fft: Boolean flag to enable FFT analysis.
            units: Unit system ("si", "pic", "db").

        Returns:
            Saves the generated plot to a file.
        """

    # Retrieve data
    var_data = numpy_data.get_data_x_t()


    # Determine the time index (x_level)
    if T > 1 or T < 0:
        return f"Invalid T value: T = {units}, valid values are between <{0}, {1}>"
    x_level = -1 if T == 1 else int(len(var_data) * T)

    # Get time axis and convert to SI units
    axis_time = numpy_data.get_time_axis()
    calculation_time = axis_time[x_level] * 1 / unit.ion.get_plasma_frequency()

    # Retrieve length axis only once (avoids redundant function calls)
    length_data = numpy_data.get_length_axis()

    axis_x, label_x = convert_x_axis(length_data, units)
    axis_z, label_z = convert_z_axis(var_data[x_level], set_data, units)


    # Construct axis labels and file name
    z_axis_name = set_data.split("_")[0]
    axis_z_name = f"{g_axis_name[z_axis_name]} [{label_z}]"
    save_file_name = f"{set_data}_hdf5_len.png"

    # **Plot Data**
    if res_type == "fft":
        # Safely extract the unit inside square brackets
        # _, _, unit_part = label_x.partition("[")    # Splits into three parts: before "[", the "[", and after "["
        # unit1, _, _ = unit_part.partition("]")      # Extracts the unit inside brackets
        # unit1 = unit1.strip()                       # Trim whitespace for safety

        description = ploting.PlotDescription(
            f"Frequency Spectrum for {set_data}; t = {round(calculation_time * 1000, 1)} ms",
            fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$", "Magnitude")
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_x, axis_z, description, False, fft_file_name)
    elif res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {set_data}; t = {round(calculation_time * 1000, 1)} ms",
            axis_z_name, "Magnitude")
        ploting.plot_histogram(axis_z, 4096, description_hist, False, save_file_name)

    elif res_type == "val":
        description = ploting.PlotDescription(f"Cut data {set_data} for t = {round(calculation_time * 1000, 3)} ms",
                                          label_x, axis_z_name)
        ploting.plot_data(axis_x, axis_z, description, False, save_file_name)
    else:
        return f"Type of data for graph: {res_type} is not implemented"


def graph_multi_time_value(numpy_data, set_data, T, res_type = "val", units="si"):
    """Generates a graph for a given dataset at a specific time with optional FFT analysis.

        Args:
            numpy_data: Object containing the dataset.
            set_data: String representing the dataset name.
            T: Time index factor (1 means last entry).
            fft: Boolean flag to enable FFT analysis.
            units: Unit system ("si", "pic", "db").

        Returns:
            Saves the generated plot to a file.
        """
    if not isinstance(numpy_data, list):
        numpy_data = [numpy_data]

    if not isinstance(set_data, list):
        set_data = [set_data]


    # Retrieve data
    var_data = [data.get_data_x_t() for data in numpy_data]
    # var_data = numpy_data.get_data_x_t()


    # Determine the time index (x_level)
    if T > 1 or T < 0:
        return f"Invalid T value: T = {units}, valid values are between <{0}, {1}>"
    x_level = -1 if T == 1 else int(len(var_data[0]) * T)

    # Get time axis and convert to SI units
    axis_time = numpy_data[0].get_time_axis()
    calculation_time = axis_time[x_level] * 1 / unit.ion.get_plasma_frequency()

    # Retrieve length axis only once (avoids redundant function calls)
    length_data = numpy_data[0].get_length_axis()

    axis_x, label_x = convert_x_axis(length_data, units)
    axis_z, label_z = zip(*[convert_z_axis(v[x_level], set_data[i], units) for i, v in enumerate(var_data)])


    # Construct axis labels and file name
    z_axis_name = set_data[0].split("_")[0]

    # print(int("abc"))
    data_labels = []
    for i, s in enumerate(set_data):
        name = s.split("_")[0]
        try:
            second_name = int(s.split("_")[1])
            # z_axis_name = z_axis_name + '_' + str(second_name)
            # print(f"z_axis_name: {name}")
            data_labels.append(f"{g_axis_name[name + '_' + str(second_name)]}")
        except:
            print("except activated")
            data_labels.append(f"{g_axis_name[name]}")

    # print(data_labels)
    axis_z_name = f"{g_axis_name[z_axis_name]} [{label_z[0]}]"
    save_file_name = f"{set_data}_hdf5_len.png"
    # title_set_data = ""
    # for s in set_data:
    #     title_set_data = title_set_data + s + ", "
    title_set_data = z_axis_name

    # **Plot Data**
    if res_type == "fft":
        # Safely extract the unit inside square brackets
        # _, _, unit_part = label_x.partition("[")    # Splits into three parts: before "[", the "[", and after "["
        # unit1, _, _ = unit_part.partition("]")      # Extracts the unit inside brackets
        # unit1 = unit1.strip()                       # Trim whitespace for safety

        description = ploting.PlotDescription(
            f"Frequency Spectrum for {title_set_data}; t = {round(calculation_time * 1000, 1)} ms",
            fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$", "Magnitude")
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_x, axis_z, description, False, fft_file_name)
    elif res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {title_set_data}; t = {round(calculation_time * 1000, 3)} ms",
            axis_z_name, "Magnitude")
        description_hist.multidata_labels(data_labels)
        ploting.plot_histogram(axis_z, 4096, description_hist, False, save_file_name)

    elif res_type == "val":
        description = ploting.PlotDescription(f"Cut data {title_set_data} for t = {round(calculation_time * 1000, 3)} ms",
                                          label_x, axis_z_name)
        ploting.plot_data(axis_x, axis_z, description, False, save_file_name)
    else:
        return f"Type of data for graph: {res_type} is not implemented"


def graph_for_pos_value(numpy_data, set_data, N, res_type = "val", units="si"):
    """Generates a graph for a given dataset at a specific time with optional FFT analysis.

        Args:
            numpy_data: Object containing the dataset.
            set_data: String representing the dataset name.
            T: Time index factor (1 means last entry).
            fft: Boolean flag to enable FFT analysis.
            units: Unit system ("si", "pic", "db").

        Returns:
            Saves the generated plot to a file.
        """

    # Retrieve data
    var_data = numpy_data.get_data_x_t()

    with open("parameters_hdf5.json", "r") as file:
        parameters = json.load(file)
    setting = read.ReadHDFSettings(parameters["folder"] + "settings.hdf")
    nx = setting.get_num_cells("x")
    Lx = setting.get_box_size("x")



    # Determine the time index (x_level)
    if N > 1 or N < 0:
        return f"Invalid N value: N = {units}, valid values are between <{0}, {1}>"
    x_level = -1 if N == 1 else int(len(var_data[0]) * N)

    var_data1d = [float(d[x_level]) for d in var_data]
    # print(f"len var1d: {len(var_data1d)}")

    # Get time axis and convert to SI units
    axis_position = numpy_data.get_length_axis()
    # calculation_time = axis_time[x_level] * 1 / unit.ion.get_plasma_frequency()
    calculation_time = axis_position[x_level] * unit.ion.get_ion_skin_depth() * Lx / nx

    # Retrieve length axis only once (avoids redundant function calls)
    # length_data = numpy_data.get_length_axis()
    time_data = numpy_data.get_time_axis()

    axis_t, label_t = convert_t_axis(time_data, units)
    axis_z, label_z = convert_z_axis(var_data1d, set_data, units)


    # Construct axis labels and file name
    z_axis_name = set_data.split("_")[0]
    axis_z_name = f"{g_axis_name[z_axis_name]} [{label_z}]"
    save_file_name = f"{set_data}_hdf5_len.png"

    # **Plot Data**
    if res_type == "fft":
        # Safely extract the unit inside square brackets
        # _, _, unit_part = label_x.partition("[")    # Splits into three parts: before "[", the "[", and after "["
        # unit1, _, _ = unit_part.partition("]")      # Extracts the unit inside brackets
        # unit1 = unit1.strip()                       # Trim whitespace for safety

        description = ploting.PlotDescription(
            f"Frequency Spectrum for {set_data}; x = {round(calculation_time, 1)} m",
            fr"$Wavenumber~~[{extract_unit_from_label(label_t)}^{{-1}}]$", "Magnitude")
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_t, axis_z, description, False, fft_file_name)
    elif res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {set_data}; x = {round(calculation_time , 1)} m",
            axis_z_name, "Magnitude")
        ploting.plot_histogram(axis_z, 4096, description_hist, False, save_file_name)

    elif res_type == "val":
        description = ploting.PlotDescription(f"Cut data {set_data} for x = {round(calculation_time, 3)} m",
                                          label_t, axis_z_name)
        ploting.plot_data(axis_t, axis_z, description, False, save_file_name)
    elif res_type == "val_ms":
        if units == "si":
            label_t_ms = label_t.replace("[s]", "[ms]")
        else:
            raise Exception(f"For type {res_type} has to be chosen SI units, units=si")

        description = ploting.PlotDescription(f"Cut data {set_data} for x = {round(calculation_time, 3)} m",
                                          label_t_ms, axis_z_name)
        axis_t_ms = unit_convert.rescale_list(axis_t,1000)
        ploting.plot_data(axis_t_ms, axis_z, description, False, save_file_name)
    else:
        return f"Type of data for graph: {res_type} is not implemented"


def graph2d(numpy_data, set_data, result_units = "db"):
    """Generate a 2D time-space graph from numpy data and save as an image."""

    # Extract raw data
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()


    print(f"plasma_frequency: {unit.ion.get_plasma_frequency()}")

    # Extract y-axis label (first part of set_data before '_')
    y_axis_name = set_data.split("_")[0]
    save_file_name = f"{set_data}_2d_time.png"

    # Convert axes based on selected units
    axis_x, label_x = convert_x_axis(length_data, result_units)
    axis_y, label_y = convert_t_axis(time_data, result_units)
    axis_z, label_z_unit = convert_z_axis(var_data, set_data, result_units)

    label_z = f"{g_axis_name[y_axis_name]} [{label_z_unit}]"

    # Define plot description
    descr3D = ploting.PlotDescription(f"Time development through space for {set_data}", label_x, label_y, label_z)
    descr3D.set_ylim(read.min_value(var_data), read.max_value(var_data))

    # Plot and save the figure
    ploting.plot3Dplane_data(axis_x, axis_y, axis_z, descr3D, False, save_file_name)


def graph2d_fft_in_time(numpy_data, set_data, result_units = "db", wire_plot: bool = False):
    """Generate a 2D time-space graph from numpy data and save as an image."""

    # Extract raw data
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()

    raw_axis_x, label_x = convert_x_axis(length_data, "si")

    axis_z = []
    for level in range(0,len(var_data)):
        raw_axis_z, label_z_unit = convert_z_axis(var_data[level], set_data, "si")
        fft_axis_x, fft_axis_z = fft_data_transform(raw_axis_x, raw_axis_z)
        axis_z.append(fft_axis_z)


    # Extract y-axis label (first part of set_data before '_')
    y_axis_name = set_data.split("_")[0]
    save_file_name = f"{set_data}_2d_fft_time.png"

    # Convert axes based on selected units
    axis_x = fft_axis_x
    axis_y, label_y = convert_t_axis(time_data, result_units)
    label_z = f"{g_axis_name[y_axis_name]} [{label_z_unit}]"

    # Define plot description
    descr3D = ploting.PlotDescription(f"Time development through space for {set_data}",
                                      fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$", label_y, label_z)
    descr3D.set_ylim(read.min_value(var_data), read.max_value(var_data))

    # Plot and save the figure
    if wire_plot:
        ploting.plot3Dwire_data(axis_x, axis_y, axis_z, descr3D, False, save_file_name)
    else:
        ploting.plot3Dplane_data(axis_x, axis_y, axis_z, descr3D, False, save_file_name)


def graph_fft2d(numpy_data, set_data, result_units = "db"):
    """Generate a 2D time-space graph from numpy data and save as an image."""

    # Extract raw data
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()

    print(f"plasma_frequency: {unit.ion.get_plasma_frequency()}")

    # Extract y-axis label (first part of set_data before '_')
    # y_axis_name = set_data.split("_")[0]
    save_file_name = f"{set_data}_2d_fft.png"

    # Convert axes based on selected units
    axis_x, label_x = convert_x_axis(length_data, result_units)
    axis_y, label_y = convert_t_axis(time_data, result_units)
    # axis_z, label_z_unit = convert_z_axis(var_data, result_units)

    # label_z = f"{g_axis_name[y_axis_name]} [{label_z_unit}]"

    # FFT Processing
    fft_results = read.fft_2d(np.array(var_data), axis_x, axis_y)
    if not fft_results:
        return  # Exit early if FFT fails

    E_fft_shifted, kx_shifted, ky_shifted = fft_results

    # xx_c, tt_c, zz_c = map(extract_half, (kx_shifted, ky_shifted, np.abs(E_fft_shifted)))
    #
    # print(f"FFT 2D Plane: {len(kx_shifted)} x {len(ky_shifted)}")
    # print(f"Processed Data Plane: {len(zz_c[0])} x {len(zz_c)}")
    #
    # # Downsampling factors (adjust as needed)
    # scale_x, scale_t, scale_z = 40, 200, 4
    #
    # # Reduce dataset size
    # xx_c1, tt_c1 = xx_c[:len(xx_c) // scale_x], tt_c[:len(tt_c) // scale_t]
    # zz_c1 = [sublist[:len(sublist) // scale_x] for sublist in zz_c[:len(zz_c) // scale_t]]


    print("PRINT FFT 2D")

        # zz = np.abs(E_fft_shifted)
    # tt = ky_shifted
    # xx = kx_shifted
    positive_semi_space_z_tdir = [sublist[len(sublist) // 2:] for sublist in np.abs(E_fft_shifted)]

    positive_semi_space_x = kx_shifted[len(kx_shifted) // 2:]
    positive_semi_space_y = ky_shifted[len(ky_shifted) // 2:]
    positive_semi_space_z = positive_semi_space_z_tdir[len(positive_semi_space_z_tdir) // 2:]

    # print(f"plane xt: {len(kx_shifted)} x {len(ky_shifted)}")
    # print(f"data plane: {len(np.abs(E_fft_shifted)[0])} x {len(np.abs(E_fft_shifted))}")
    # print("planes after cut off")
    # print(f"plane xt: {len(xx_c)} x {len(tt_c)}")
    # print(f"data plane: {len(zz_c[0])} x {len(zz_c)}")

    scale_x = 40
    scale_t = 200
    scale_z = 4

    res_axis_x = positive_semi_space_x[:len(positive_semi_space_x) // scale_x]
    res_axis_y = positive_semi_space_y[:len(positive_semi_space_y) // scale_t]
    # zz_mem = [sublist[:len(sublist) // scale_x] for sublist in zz_c]
    res_axis_z = positive_semi_space_z[:len(positive_semi_space_z) // scale_t]

    descr3D = ploting.PlotDescription(r"$FFT~2D~result~of~E_x$", fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$",
                                          fr"$Frequency~~[{extract_unit_from_label(label_y)}]$",
                                          "Magnitude")
        # descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    ploting.plot3Dplane_data(res_axis_x, res_axis_y, res_axis_z, descr3D, False,
        "fft_" + save_file_name, scale_z)


def graph_energy(numpy_data, set_data, res_type = "val", units="si"):

    energy_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    axis_t, label_t = convert_t_axis(time_data, units)
    axis_z, label_z = convert_z_axis(energy_data, set_data, units)

    z_axis_name = set_data

    if res_type == "val":
        axis_z_name = f"{g_axis_name[z_axis_name]} [{label_z[0]}]"
        description = ploting.PlotDescription(f"Time development {g_axis_name[set_data]}", label_t, axis_z_name)
        ploting.plot_data(axis_t, axis_z, description, False, "field")
    else:
        return f"Type of data for graph: {res_type} is not implemented"



# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
#                                   HELP FUNCTIONS
def convert_x_axis(axis_x, units="si"):
    with open("parameters_hdf5.json", "r") as file:
        parameters = json.load(file)
    setting = read.ReadHDFSettings(parameters["folder"] + "settings.hdf")
    nx = setting.get_num_cells("x")
    Lx = setting.get_box_size("x")

    # Lx = 0.5
    # nx = 4096

    match units:
        case "pic":
            unit_x = r"$Length~~[isd]$"
            return [axis_x, unit_x]
        case "si":
            axis_x_SI = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * Lx / nx)
            unit_x = r"$Length~~[m]$"
            return [axis_x_SI, unit_x]
        case "db":
            axis_x_SI = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * Lx / nx)
            axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1 / unit.debye_len)
            unit_x = r"$Length~~[\lambda_{D}]$"
            return [axis_x_DB, unit_x]
        case _:
            return f"Invalid unit type: {units}"


def convert_t_axis(axis_time, units="si"):
    match units:
        case "pic":
            unit_y = r"$Time~~[1/\omega_{pi}]$"
            return [axis_time, unit_y]
        case "si":
            axis_time_SI = unit_convert.rescale_list(axis_time, 1 / unit.ion.get_plasma_frequency())
            unit_y = r"$Time~~[s]$"
            return [axis_time_SI, unit_y]
        case "db":
            axis_time_SI = unit_convert.rescale_list(axis_time, 1 / unit.ion.get_plasma_frequency())
            axis_time_OM = unit_convert.rescale_list(axis_time_SI, unit.electron.get_plasma_frequency())
            unit_y = r"$Time~~[1/\omega_{pe}]$"
            return [axis_time_OM, unit_y]
        case _:
            return f"Invalid unit type: {units}"


def convert_z_axis(axis_y, set_data, units="si"):
    variable = set_data.split("_")[0]

    scale_factors = {
        "T": unit.c1.b_field_const,
        "m/s": unit.c1.vel_const,
        "V/m": unit.c1.e_field_const,
        "J": unit.c1.energy_const,
        "A/m": unit.c1.charge_const,
    }
    # print(f"convert_z_axis.variable: {variable}")
    match units:
        case "pic":
            axis_z_data = axis_y
            unit_z = g_axis_pic_units[variable]
            # print(f"unit_y PIC: {unit_z}")
            return [axis_z_data, unit_z]
        case "si" | "db":
            unit_z = g_axis_si_units[variable]
            scale = scale_factors.get(unit_z)
            if scale:
                axis_z_data = unit_convert.rescale_list(axis_y, scale)
            else:
                axis_z_data = axis_y  # fallback to original if unit not found

            # print(f"unit_z SI: {unit_z}")
            return [axis_z_data, unit_z]
        case _:
            return f"Invalid unit type: {units}"


def fft_data_transform(dataX, dataY):
    aa = 50
    step = dataX[aa] - dataX[aa - 1]

    # Perform FFT
    fft_result = np.fft.fft(dataY)
    frequencies = np.fft.fftfreq(len(dataY), d=step)

    # Get magnitude spectrum (optional)
    magnitude = np.abs(fft_result)


    return [frequencies[1:len(frequencies) // 2], magnitude[1:len(magnitude) // 2]]


def extract_unit_from_label(label):
    # Safely extract the unit inside square brackets
    _, _, unit_part = label.partition("[")  # Splits into three parts: before "[", the "[", and after "["
    unit1, _, _ = unit_part.partition("]")  # Extracts the unit inside brackets
    unit1 = unit1.strip()  # Trim whitespace for safety

    return unit1


def extract_half(data):
    return data[len(data) // 2:]