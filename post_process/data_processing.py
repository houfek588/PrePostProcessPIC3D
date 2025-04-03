
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
g_axis_name = {
    "Efield": "Electric field ",
    "Bfield": "Magnetic field []",
    "rhoe": "Electron density []",
    "rhoi": "Ion density []",
    "Je": "Electron current density []",
    "Ji": "Ion current density []",
}

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
        # axis_time = []
        # for i in range(0, len(self.data)):
        #     axis_time.append(i)

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

    print(f"Data for analysis: {set_data}")

    numpy_data = ReadNMPFileData(set_data)
    # data = numpy_data.get_data_x_t()

    # print(f"len T: {len(data)}")
    # print(f"len L:{len(data[0])}")
    # print("Numpy data loaded")



    # data_si = unit_convert.rescale_list_of_lists(data, unit.c1.e_field_const)
    #
    # x_level = -1
    #
    # axis_time = numpy_data.get_time_axis()
    # print(f"axis_time: {axis_time}")
    # print(f"axis_time[-1]/0.0001: {axis_time[-1]/0.0001}")
    #
    # axis_x = numpy_data.get_length_axis()
    # axis_x_SI = unit_convert.rescale_list(axis_x, unit.ion.get_ion_skin_depth() * 0.5/4096)
    #
    # print(f"plasma_frequency: {unit.ion.get_plasma_frequency()}")
    # axis_time_SI = unit_convert.rescale_list(axis_time, 1 / unit.ion.get_plasma_frequency())
    # axis_time_SIms = unit_convert.rescale_list(axis_time_SI, 1000)
    #
    # # debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)
    # axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1 / unit.debye_len)
    # axis_time_OM = unit_convert.rescale_list(axis_time_SI, unit.electron.get_plasma_frequency())
    #
    # print(f"length DB max: {axis_x_DB[-1]}")
    # print(f"time OM max: {axis_time_OM[-1]}")

    # graph parameters for processing
    # # x_level = -1
    # save_file_name = set_data + "_" + "hdf5_len.png"
    # val1 = data_si[x_level]
    #
    # y_axis_name = set_data.split("_")[0]



    # graph_for_time_value(numpy_data, set_data, 1)
    # graph_for_time_value(numpy_data, set_data, 1, True)
    # graph2d(numpy_data, set_data)
    # graph2d_fft_in_time(numpy_data, set_data)
    graph_fft2d(numpy_data, set_data)


        # plot data directly
    # descr11 = ploting.PlotDescription(f"Length data {set_data} for t = {round(axis_time_SIms[x_level], 3)} ms",
    #                                       "Length [m]", g_axis_name[y_axis_name])
    #     # descr11.set_ylim(read.min_value(data_x_t) * 0.95, read.max_value(data_x_t) * 1.05)
    # ploting.plot_data(axis_x_SI, val1, descr11, False, save_file_name)
    #
    #     # plot data with FFT
    #     # if enable_fft:
    # descr22 = ploting.PlotDescription(
    #             f"Frequency Spectrum for {set_data}; t = {round(axis_time_SIms[x_level], 1)} ms",
    #             "Wavenumber [m-1]",
    #             "Magnitude")
    # fft_file_name = read.add_suffix(save_file_name, "_fft.")
    # ploting.plot_fft(axis_x_SI, val1, descr22, False, fft_file_name)

    # save = bool(strtobool(parameters["visualization_parameters"]["plot_2D"]["save_to_file"]))
    # save_file_name = variable_name + "_" + parameters["visualization_parameters"]["plot_2D"]["file_name"]



def graph_for_time_value(numpy_data, set_data, T, fft: bool = False, units="si"):
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
    axis_z, label_z = convert_z_axis(var_data[x_level], units)


    # Construct axis labels and file name
    y_axis_name = set_data.split("_")[0]
    axis_y_name = f"{g_axis_name[y_axis_name]} [{label_z}]"
    save_file_name = f"{set_data}_hdf5_len.png"

    # **Plot Data**
    if fft:
        # Safely extract the unit inside square brackets
        # _, _, unit_part = label_x.partition("[")    # Splits into three parts: before "[", the "[", and after "["
        # unit1, _, _ = unit_part.partition("]")      # Extracts the unit inside brackets
        # unit1 = unit1.strip()                       # Trim whitespace for safety

        description = ploting.PlotDescription(
            f"Frequency Spectrum for {set_data}; t = {round(calculation_time * 1000, 1)} ms",
            fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$", "Magnitude")
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_x, axis_z, description, False, fft_file_name)
    else:
        description = ploting.PlotDescription(f"Cut data {set_data} for t = {round(calculation_time * 1000, 3)} ms",
                                          label_x, axis_y_name)
        ploting.plot_data(axis_x, axis_z, description, False, save_file_name)


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
    axis_z, label_z_unit = convert_z_axis(var_data, result_units)

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
        raw_axis_z, label_z_unit = convert_z_axis(var_data[level], "si")
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
    y_axis_name = set_data.split("_")[0]
    save_file_name = f"{set_data}_2d_time.png"

    # Convert axes based on selected units
    axis_x, label_x = convert_x_axis(length_data, result_units)
    axis_y, label_y = convert_t_axis(time_data, result_units)
    axis_z, label_z_unit = convert_z_axis(var_data, result_units)

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
    zz = [sublist[len(sublist) // 2:] for sublist in np.abs(E_fft_shifted)]
        # zz = np.abs(E_fft_shifted)
    tt = ky_shifted
    xx = kx_shifted

    xx_c = xx[len(xx) // 2:]
    tt_c = tt[len(tt) // 2:]
    zz_c = zz[len(zz) // 2:]

    print(f"plane xt: {len(kx_shifted)} x {len(ky_shifted)}")
    print(f"data plane: {len(np.abs(E_fft_shifted)[0])} x {len(np.abs(E_fft_shifted))}")
    print("planes after cut off")
    print(f"plane xt: {len(xx_c)} x {len(tt_c)}")
    print(f"data plane: {len(zz_c[0])} x {len(zz_c)}")

    scale_x = 40
    scale_t = 200
    scale_z = 4

    xx_c1 = xx_c[:len(xx_c) // scale_x]
    tt_c1 = tt_c[:len(tt_c) // scale_t]
    zz_mem = [sublist[:len(sublist) // scale_x] for sublist in zz_c]
    zz_c1 = zz_c[:len(zz_c) // scale_t]

    descr3D = ploting.PlotDescription(r"$FFT~2D~result~of~E_x$", fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$",
                                          fr"$Frequency~~[{extract_unit_from_label(label_y)}]$",
                                          "Magnitude")
        # descr3D.set_ylim(read.min_value(data_x_t), read.max_value(data_x_t))
    ploting.plot3Dplane_data(xx_c1, tt_c1, zz_c1, descr3D, False,
        "fft_" + save_file_name, scale_z)






# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
#                                   HELP FUNCTIONS
def convert_x_axis(axis_x, units="si"):
    Lx = 0.5
    nx = 4096

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


def convert_z_axis(axis_y, units="si"):

    match units:
        case "pic":
            axis_y_data = axis_y
            unit_y = "om_i"
            return [axis_y_data, unit_y]
        case "si":
            data_si = unit_convert.rescale_list(axis_y, unit.c1.e_field_const)
            axis_y_data = data_si
            unit_y = "V/m"
            return [axis_y_data, unit_y]
        case "db":
            data_si = unit_convert.rescale_list(axis_y, unit.c1.e_field_const)
            axis_y_data = data_si
            unit_y = "V/m"
            return [axis_y_data, unit_y]
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