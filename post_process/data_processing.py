"""
data_processing.py - Zpracování mezipaměťových NumPy dat (.npy), přepočty os/jednotek a generování grafů.
"""

import math
import os
import numpy as np

try:
    from . import ploting
    from . import unit_convert
    from . import read_files as read
    from .config_manager import load_config, load_plot_config, to_bool, get_cache_dir
    from .vtk_processing import read_step_size
    from .plasma_units import get_default_units
except ImportError:
    import ploting
    import unit_convert
    import read_files as read
    from config_manager import load_config, load_plot_config, to_bool, get_cache_dir
    from vtk_processing import read_step_size
    from plasma_units import get_default_units


# Definice klíčů konfigurací (odděleno od win.py pro zamezení cyklickým importům)
SINGLE_CONFIG_KEYS = ("Value", "FFT", "Histogram", "Value2D", "FFT2D", "FFT in time")
MULTI_CONFIG_KEYS = ("Value", "FFT", "Histogram")

# Strukturovaný slovník fyzikálních veličin
QUANTITY_DEFINITIONS = {
    "Efield": {"name": "Electric field", "pic_unit": "om_i", "si_unit": "V/m", "type": "field", "weight": 1.0},
    "Bfield": {"name": "Magnetic field", "pic_unit": "v_A/c", "si_unit": "T", "type": "field", "weight": 1.0},
    "rhoe0": {"name": "Electron density", "pic_unit": "4pi", "si_unit": "1", "type": "field", "weight": 1.0},
    "rhoi1": {"name": "Ion density", "pic_unit": "4pi", "si_unit": "1", "type": "field", "weight": 1.0},
    "rhoe2": {"name": "Electron beam density", "pic_unit": "4pi", "si_unit": "1", "type": "field", "weight": 1.0},
    "Je": {"name": "Electron current density", "pic_unit": "A/m2", "si_unit": "A/m2", "type": "field", "weight": 1.0},
    "Ji": {"name": "Ion current density", "pic_unit": "A/m2", "si_unit": "A/m2", "type": "field", "weight": 1.0},
    "velocity": {"name": "Velocity", "pic_unit": "c", "si_unit": "m/s", "type": "part", "weight": 1.0},
    "velocity_1": {"name": "Electron velocity", "pic_unit": "c", "si_unit": "m/s", "type": "part", "weight": 0.99},
    "velocity_2": {"name": "Ion velocity", "pic_unit": "c", "si_unit": "m/s", "type": "part", "weight": 1.0},
    "velocity_3": {"name": "Electron beam velocity", "pic_unit": "c", "si_unit": "m/s", "type": "part", "weight": 0.5},
    "energy": {"name": "Energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_kin": {"name": "Kinetic energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_ele": {"name": "Electric field energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_kin0": {"name": "Electron kin. energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_kin1": {"name": "Ion kin. energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_kin2": {"name": "Electron beam kin. energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
    "energy_total": {"name": "Total energy", "pic_unit": "mi*c^2", "si_unit": "J/m3", "type": "field", "weight": 1.0},
}

# Zpětně kompatibilní mapovací slovníky
g_keys = list(QUANTITY_DEFINITIONS.keys())
g_names = [v["name"] for v in QUANTITY_DEFINITIONS.values()]
g_pic_units = [v["pic_unit"] for v in QUANTITY_DEFINITIONS.values()]
g_si_units = [v["si_unit"] for v in QUANTITY_DEFINITIONS.values()]
g_type_data = [v["type"] for v in QUANTITY_DEFINITIONS.values()]
g_weights = [v["weight"] for v in QUANTITY_DEFINITIONS.values()]

g_axis_name = {k: v["name"] for k, v in QUANTITY_DEFINITIONS.items()}
g_axis_si_units = {k: v["si_unit"] for k, v in QUANTITY_DEFINITIONS.items()}
g_axis_pic_units = {k: v["pic_unit"] for k, v in QUANTITY_DEFINITIONS.items()}
g_axis_type = {k: v["type"] for k, v in QUANTITY_DEFINITIONS.items()}
g_hist_weights = {k: v["weight"] for k, v in QUANTITY_DEFINITIONS.items()}


class ReadNMPFileData:
    """Načítá 1D nebo 2D data z mezipaměti script_data/ jako NumPy pole."""

    def __init__(self, set_data: str):
        self.set_data = set_data
        cache_dir = get_cache_dir()
        file_2d = os.path.join(cache_dir, f"{set_data}_data2D.npy")
        file_1d = os.path.join(cache_dir, f"{set_data}_data1D.npy")


        if os.path.exists(file_2d):
            self.data = np.load(file_2d)
            self.data_dim = 2
        elif os.path.exists(file_1d):
            self.data = np.load(file_1d)
            self.data_dim = 1
        else:
            raise FileNotFoundError(f"Mezipaměťová data pro '{set_data}' nebyla nalezena (ani 1D, ani 2D).")

        z_axis_name = set_data.split("_")[0]
        # Pokud je v názvu i číslo (např. velocity_1), zkontrolujeme plný název
        category_full = "_".join(set_data.split("_")[:2]) if len(set_data.split("_")) > 2 else z_axis_name
        if category_full in g_axis_type:
            self.type = g_axis_type[category_full]
        elif z_axis_name in g_axis_type:
            self.type = g_axis_type[z_axis_name]
        else:
            self.type = "field"

    def get_data_dimension(self) -> int:
        return self.data_dim

    def get_data_x_t(self) -> np.ndarray:
        return self.data

    def get_time_axis(self) -> list:
        cfg = load_config()
        folder = cfg.get("result_folder", "")
        settings_path = os.path.join(folder, "settings.hdf")
        setting = read.ReadHDFSettings(settings_path)
        dt = setting.get_time_step_size()
        step = read_step_size(self.type)

        n_steps = len(self.data)
        return [i * step * dt for i in range(n_steps)]

    def get_length_axis(self):
        if self.data_dim == 1:
            return "This function is not available for 1D data"
        return list(range(len(self.data[0])))


class ReadPlotConfig:
    """Čte konfiguraci vykreslování (plot_config.json) bez závislosti na GUI."""

    def __init__(self, config_file: str = "plot_config.json"):
        self.cfg = load_plot_config(config_file)

    def get_single_value(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[0], False))

    def get_single_fft(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[1], False))

    def get_single_hist(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[2], False))

    def get_single_value2d(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[3], False))

    def get_single_fft2d(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[4], False))

    def get_single_fft_in_time(self) -> bool:
        return to_bool(self.cfg.get("single", {}).get(SINGLE_CONFIG_KEYS[5], False))

    def get_multi_value(self) -> bool:
        return to_bool(self.cfg.get("multi", {}).get(MULTI_CONFIG_KEYS[0], False))

    def get_multi_fft(self) -> bool:
        return to_bool(self.cfg.get("multi", {}).get(MULTI_CONFIG_KEYS[1], False))

    def get_multi_hist(self) -> bool:
        return to_bool(self.cfg.get("multi", {}).get(MULTI_CONFIG_KEYS[2], False))

    def __str__(self):
        return str(self.cfg)


def single_result_analysis(set_data: str):
    """Zpracuje a vykreslí samostatnou veličinu na základě plot_config.json."""
    print(f"Spouštím analýzu samostatné veličiny: {set_data}")

    numpy_data = ReadNMPFileData(set_data)
    plot_cfg = ReadPlotConfig("plot_config.json")

    if numpy_data.get_data_dimension() == 1:
        if plot_cfg.get_single_value():
            graph_energy(numpy_data, set_data, "val", "db")
        if plot_cfg.get_single_fft():
            graph_energy(numpy_data, set_data, "fft", "db")
        if plot_cfg.get_single_hist():
            graph_energy(numpy_data, set_data, "hist", "db")
    else:
        if plot_cfg.get_single_value():
            graph_for_pos_value(numpy_data, set_data, 0.5, "val", "db")
        if plot_cfg.get_single_fft():
            graph_for_pos_value(numpy_data, set_data, 0.5, "fft", "db")
        if plot_cfg.get_single_hist():
            graph_for_pos_value(numpy_data, set_data, 0.5, "hist", "db")
        if plot_cfg.get_single_value2d():
            graph2d(numpy_data, set_data, "db")
        if plot_cfg.get_single_fft2d():
            graph_fft2d(numpy_data, set_data)
        if plot_cfg.get_single_fft_in_time():
            graph2d_fft_in_time(numpy_data, set_data)

    print(f"Analýza pro '{set_data}' dokončena.")


def multi_result_analysis(set_data: list):
    """Zpracuje a vykreslí srovnávací grafy pro více veličin najednou."""
    print(f"Spouštím multi-analýzu pro: {set_data}")

    if not set_data:
        print("Seznam multi_var je prázdný, přeskakuji.")
        return "The list is empty"

    numpy_data = [ReadNMPFileData(name) for name in set_data]
    plot_cfg = ReadPlotConfig("plot_config.json")

    if plot_cfg.get_multi_value():
        graph_multi_time_value(numpy_data, set_data, 0, "val", "pic", user_id="_T0")
        graph_multi_time_value(numpy_data, set_data, 1, "val", "pic", user_id="_T1")

    if plot_cfg.get_multi_fft():
        graph_multi_time_value(numpy_data, set_data, 0, "fft", "pic", user_id="_T0")
        graph_multi_time_value(numpy_data, set_data, 1, "fft", "pic", user_id="_T1")

    if plot_cfg.get_multi_hist():
        graph_multi_time_value(numpy_data, set_data, 0, "hist", "db", user_id="_T0", weights_on=True, merge=False)
        graph_multi_time_value(numpy_data, set_data, 1, "hist", "db", user_id="_T1", weights_on=True, merge=False)

    print("Multi-analýza dokončena.")
    return "Multi Data was processed"


def graph_for_time_value(numpy_data, set_data, T, res_type="val", units="si", user_id=""):
    """Vykreslí prostorový řez veličinou v daném čase T (kde T in <0, 1>)."""
    var_data = numpy_data.get_data_x_t()
    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    if not (0 <= T <= 1):
        return f"Neplatná hodnota T = {T}, povolený rozsah je <0, 1>"

    x_level = -1 if T == 1 else int(len(var_data) * T)

    axis_time = numpy_data.get_time_axis()
    unit = get_default_units()
    om_pi = unit.ion.get_plasma_frequency()
    calculation_time = (axis_time[x_level] / om_pi) if om_pi > 0 else axis_time[x_level]

    length_data = numpy_data.get_length_axis()
    axis_x, label_x = convert_x_axis(length_data, units)
    axis_z, label_z = convert_z_axis(var_data[x_level], set_data, units)

    z_axis_name = set_data.split("_")[0]
    axis_name_label = g_axis_name.get(z_axis_name, z_axis_name)
    axis_z_name = f"{axis_name_label} [{label_z}]"
    save_file_name = f"{set_data}_lenDevel{user_id}.png"

    if res_type == "fft":
        description = ploting.PlotDescription(
            f"Frequency Spectrum for {set_data}; t = {round(calculation_time * 1000, 1)} ms",
            fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$",
            "Magnitude",
        )
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_x, axis_z, description, save_graph, os.path.join(img_folder, fft_file_name))
    elif res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {set_data}; t = {round(calculation_time * 1000, 1)} ms",
            axis_z_name,
            "f(u)",
        )
        ploting.plot_histogram(axis_z, 4096, description_hist, [1.0] * len(axis_z), save_graph, os.path.join(img_folder, save_file_name))
    elif res_type == "val":
        description = ploting.PlotDescription(
            f"Cut data {set_data} for t = {round(calculation_time * 1000, 3)} ms",
            label_x,
            axis_z_name,
        )
        ploting.plot_data(axis_x, axis_z, description, save_graph, os.path.join(img_folder, save_file_name))
    else:
        return f"Typ grafu '{res_type}' není implementován"


def graph_multi_time_value(numpy_data, set_data, T, res_type="val", units="si", user_id="", weights_on=True, merge=False):
    """Vykreslí srovnávací graf pro více veličin v daném čase T."""
    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    axis_x = []
    axis_z = []
    data_labels = []
    weights = []

    unit = get_default_units()
    for idx, d in enumerate(numpy_data):
        var_data = d.get_data_x_t()
        x_level = -1 if T == 1 else int(len(var_data) * T)

        axis_time = d.get_time_axis()
        om_pi = unit.ion.get_plasma_frequency()
        calculation_time = (axis_time[x_level] / om_pi) if om_pi > 0 else axis_time[x_level]

        time_om = axis_time[x_level] * unit.electron.get_plasma_frequency() / (2.0 * math.pi)


        length_data = d.get_length_axis()
        ax_x, label_x = convert_x_axis(length_data, units)
        ax_z, label_z = convert_z_axis(var_data[x_level], set_data[idx], units)

        axis_x.append(ax_x)
        axis_z.append(ax_z)

        category = set_data[idx].split("_")[0]
        full_cat = "_".join(set_data[idx].split("_")[:2]) if len(set_data[idx].split("_")) > 2 else category
        name_label = g_axis_name.get(full_cat, g_axis_name.get(category, set_data[idx]))
        data_labels.append(name_label)

        if weights_on:
            w_val = g_hist_weights.get(full_cat, g_hist_weights.get(category, 1.0))
            weights.append(np.ones(len(ax_z)) * w_val)
        else:
            weights.append(np.ones(len(ax_z)))

    title_set_data = " & ".join(set_data)
    axis_z_name = f"{label_z}"
    save_file_name = f"multi_{title_set_data}_t{T}{user_id}.png"

    if res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {title_set_data}; t = {round(time_om, 2)} [2pi/om_pe]",
            axis_z_name,
            "f(u)",
        )
        description_hist.multidata_labels(data_labels)

        if merge:
            merged = np.concatenate(axis_z, axis=None)
            m_weights = np.ones(len(merged))
            save_file_name = "merged_" + save_file_name
            ploting.plot_histogram(merged, 4096, description_hist, m_weights, save_graph, os.path.join(img_folder, save_file_name))
        else:
            ploting.plot_histogram(axis_z, 4096, description_hist, weights, save_graph, os.path.join(img_folder, save_file_name))
    elif res_type == "val":
        description = ploting.PlotDescription(
            f"Cut data {title_set_data} for t = {round(calculation_time * 1000, 3)} ms",
            label_x,
            axis_z_name,
        )
        description.multidata_labels(data_labels)
        ploting.plot_data(axis_x[0], axis_z, description, save_graph, os.path.join(img_folder, save_file_name))


def graph_for_pos_value(numpy_data, set_data, N, res_type="val", units="si", user_id=""):
    """Vykreslí časový vývoj v dané prostorové poloze N (kde N in <0, 1>)."""
    var_data = numpy_data.get_data_x_t()
    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    settings_path = os.path.join(cfg.get("result_folder", ""), "settings.hdf")
    setting = read.ReadHDFSettings(settings_path)
    nx = setting.get_num_cells("x")
    Lx = setting.get_box_size("x")

    if not (0 <= N <= 1):
        return f"Neplatná hodnota N = {N}, povolený rozsah je <0, 1>"

    x_level = -1 if N == 1 else int(len(var_data[0]) * N)
    var_data1d = var_data[:, x_level] if isinstance(var_data, np.ndarray) else [float(d[x_level]) for d in var_data]

    axis_position = numpy_data.get_length_axis()
    unit = get_default_units()
    if units == "si":
        calc_pos = (axis_position[x_level] * unit.ion.get_ion_skin_depth() * Lx / nx) if nx > 0 else axis_position[x_level]

        title_unit = f"{round(calc_pos, 2)} m"
    elif units == "db":
        ax_x, lbl_x = convert_x_axis(axis_position, "db")
        calc_pos = ax_x[x_level]
        title_unit = f"{round(calc_pos, 0)} lambda_D"
    else:
        calc_pos = axis_position[x_level]
        title_unit = f"{round(calc_pos, 0)} isd"

    time_data = numpy_data.get_time_axis()
    axis_t, label_t = convert_t_axis(time_data, units)
    axis_z, label_z = convert_z_axis(var_data1d, set_data, units)

    z_axis_name = set_data.split("_")[0]
    axis_name_label = g_axis_name.get(z_axis_name, z_axis_name)
    axis_z_name = f"{axis_name_label} [{label_z}]"
    save_file_name = f"{set_data}_timeDevel{user_id}.png"

    if res_type == "fft":
        description = ploting.PlotDescription(
            f"Frequency Spectrum for {set_data}; x = {title_unit}",
            fr"$Frequency~~[{extract_unit_from_label(label_t)}^{{-1}}]$",
            "Magnitude",
        )
        fft_file_name = read.add_suffix(save_file_name, "_fft.")
        ploting.plot_fft(axis_t, axis_z, description, save_graph, os.path.join(img_folder, fft_file_name))
    elif res_type == "hist":
        save_file_name = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {set_data}; x = {title_unit}",
            axis_z_name,
            "Magnitude",
        )
        ploting.plot_histogram(axis_z, 4096, description_hist, [1.0] * len(axis_z), save_graph, os.path.join(img_folder, save_file_name))
    elif res_type == "val":
        description = ploting.PlotDescription(
            f"Cut data {set_data} for x = {title_unit}",
            label_t,
            axis_z_name,
        )
        ploting.plot_data(axis_t, axis_z, description, save_graph, os.path.join(img_folder, save_file_name))


def graph2d(numpy_data, set_data, result_units="db", user_id=""):
    """Vykreslí 2D časoprostorový vývoj veličiny."""
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()

    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    y_axis_name = set_data.split("_")[0]
    save_file_name = f"{set_data}_2d_timeDevel{user_id}.png"

    axis_x, label_x = convert_x_axis(length_data, result_units)
    axis_y, label_y = convert_t_axis(time_data, result_units)
    axis_z, label_z_unit = convert_z_axis(var_data, set_data, result_units)

    label_z = f"{g_axis_name.get(y_axis_name, y_axis_name)} [{label_z_unit}]"
    descr3D = ploting.PlotDescription(f"Time development through space for {set_data}", label_x, label_y, label_z)

    ploting.plot3Dplane_data(axis_x, axis_y, axis_z, descr3D, save_graph, os.path.join(img_folder, save_file_name))


def graph2d_fft_in_time(numpy_data, set_data, result_units="db", wire_plot: bool = False, user_id=""):
    """Vykreslí časový vývoj prostorového vlnového spektra (k vs. t)."""
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()

    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    raw_axis_x, label_x = convert_x_axis(length_data, result_units)

    axis_z = []
    fft_axis_x = None
    for level in range(len(var_data)):
        raw_axis_z, _ = convert_z_axis(var_data[level], set_data, result_units)
        fft_axis_x, fft_axis_z = fft_data_transform(raw_axis_x, raw_axis_z)
        axis_z.append(fft_axis_z)

    save_file_name = f"{set_data}_2d_freqDevel{user_id}.png"
    axis_x = fft_axis_x if fft_axis_x is not None else []
    axis_y, label_y = convert_t_axis(time_data, result_units)
    label_z = "Spectral energy [(V/m)·m]"

    descr3D = ploting.PlotDescription(
        f"Time development of wave spectrum for {set_data}",
        fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$",
        label_y,
        label_z,
    )

    if wire_plot:
        ploting.plot3Dwire_data(axis_x, axis_y, axis_z, descr3D, save_graph, os.path.join(img_folder, save_file_name))
    else:
        ploting.plot3Dplane_data(axis_x, axis_y, axis_z, descr3D, save_graph, os.path.join(img_folder, save_file_name))


def graph_fft2d(numpy_data, set_data, result_units="db", user_id=""):
    """Provede 2D FFT v prostoru a čase a vykreslí (k, omega) disperzní diagram."""
    var_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    length_data = numpy_data.get_length_axis()

    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))

    save_file_name = f"{set_data}_2d_fft{user_id}.png"

    axis_x, label_x = convert_x_axis(length_data, result_units)
    axis_y, label_y = convert_t_axis(time_data, result_units)

    fft_results = read.fft_2d(np.asarray(var_data), axis_x, axis_y)
    if not fft_results:
        return

    E_fft_shifted, kx_shifted, ky_shifted = fft_results

    # Kladný poloprostor pro frekvence a vlnová čísla
    mag_fft = np.abs(E_fft_shifted)
    ny_half = len(ky_shifted) // 2
    nx_half = len(kx_shifted) // 2

    positive_semi_space_x = kx_shifted[nx_half:]
    positive_semi_space_y = ky_shifted[ny_half:]
    positive_semi_space_z = mag_fft[ny_half:, nx_half:]

    # Adaptivní zobrazení rezonanční oblasti
    scale_x = 0.05
    scale_t = 0.02

    end_x = max(2, int(len(positive_semi_space_x) * scale_x))
    end_y = max(2, int(len(positive_semi_space_y) * scale_t))

    res_axis_x = positive_semi_space_x[:end_x]
    res_axis_y = positive_semi_space_y[:end_y]
    res_axis_z = positive_semi_space_z[:end_y, :end_x]

    label_z = "Spectral energy [(V/m)·s·m]"
    unit_y = r"$Frequency~~[\omega_{pe}]$"
    descr3D = ploting.PlotDescription(
        fr"$FFT~2D~result~of~{set_data}$",
        fr"$Wavenumber~~[{extract_unit_from_label(label_x)}^{{-1}}]$",
        unit_y,
        label_z,
    )
    ploting.plot3Dplane_data(res_axis_x, res_axis_y, res_axis_z, descr3D, save_graph, os.path.join(img_folder, save_file_name), max_bar_val=50)


def graph_energy(numpy_data, set_data, res_type="val", units="si", user_id=""):
    """Vykreslí časový vývoj integrální energie."""
    energy_data = numpy_data.get_data_x_t()
    time_data = numpy_data.get_time_axis()
    axis_t, label_t = convert_t_axis(time_data, units)
    axis_z, label_z = convert_z_axis(energy_data, set_data, units)

    cfg = load_config()
    img_folder = cfg.get("output_folder", "")
    save_graph = to_bool(cfg.get("save_graphs", True))
    save_file_name = f"{set_data}{user_id}.png"

    if res_type == "val":
        name_label = g_axis_name.get(set_data, set_data)
        axis_z_name = f"{name_label} [{label_z}]"
        description = ploting.PlotDescription(f"Time development of {name_label}", label_t, axis_z_name)
        ploting.plot_data(axis_t, axis_z, description, save_graph, os.path.join(img_folder, save_file_name))
    else:
        raise ValueError(f"Typ grafu '{res_type}' není pro energii implementován")


# --------------------------------------------------------------------------------------------
# POMOCNÉ FUNKCE PŘEPOČTU OS A JEDNOTEK
# --------------------------------------------------------------------------------------------

def convert_x_axis(axis_x, units="si"):
    cfg = load_config()
    settings_path = os.path.join(cfg.get("result_folder", ""), "settings.hdf")
    setting = read.ReadHDFSettings(settings_path)
    nx = setting.get_num_cells("x")
    Lx = setting.get_box_size("x")

    unit = get_default_units()
    d_i = unit.ion.get_ion_skin_depth()
    debye_len = unit.debye_len
    dx_si = (d_i * Lx / nx) if nx > 0 else 1.0

    if units == "pic":
        return [axis_x, r"$Length~~[d_i]$"]
    elif units == "si":
        axis_x_SI = unit_convert.rescale_list(axis_x, dx_si)
        return [axis_x_SI, r"$Length~~[m]$"]
    elif units == "db":
        axis_x_SI = unit_convert.rescale_list(axis_x, dx_si)
        axis_x_DB = unit_convert.rescale_list(axis_x_SI, 1.0 / debye_len) if debye_len > 0 else axis_x_SI
        return [axis_x_DB, r"$Length~~[\lambda_{D}]$"]
    else:
        return [axis_x, f"Invalid unit type: {units}"]


def convert_t_axis(axis_time, units="si"):
    unit = get_default_units()
    om_pi = unit.ion.get_plasma_frequency()
    om_pe = unit.electron.get_plasma_frequency()

    if units == "pic":
        return [axis_time, r"$Time~~[1/\omega_{pi}]$"]
    elif units == "si":
        t_scale = (1.0 / om_pi) if om_pi > 0 else 1.0
        axis_time_SI = unit_convert.rescale_list(axis_time, t_scale)
        return [axis_time_SI, r"$Time~~[s]$"]
    elif units == "db":
        t_scale = (1.0 / om_pi) if om_pi > 0 else 1.0
        axis_time_SI = unit_convert.rescale_list(axis_time, t_scale)
        pe_factor = (om_pe / (2.0 * math.pi)) if om_pe > 0 else 1.0
        axis_time_OM = unit_convert.rescale_list(axis_time_SI, pe_factor)
        return [axis_time_OM, r"$Time~~[2\pi/\omega_{pe}]$"]
    else:
        return [axis_time, f"Invalid unit type: {units}"]


def convert_z_axis(axis_y, set_data, units="si"):
    variable = set_data.split("_")[0]
    category_full = "_".join(set_data.split("_")[:2]) if len(set_data.split("_")) > 2 else variable

    unit = get_default_units()
    scale_factors = {
        "T": getattr(unit.c1, "b_field_const", 1.0),
        "m/s": getattr(unit.c1, "vel_const", unit.const_c),
        "V/m": getattr(unit.c1, "e_field_const", 1.0),
        "J": getattr(unit.c1, "energy_const", 1.0),
        "J/m3": getattr(unit.c1, "energy_const", 1.0),
        "A/m": getattr(unit.c1, "charge_const", 1.0),
        "A/m2": getattr(unit.c1, "charge_const", 1.0),
    }

    if units == "pic":
        unit_z = g_axis_pic_units.get(category_full, g_axis_pic_units.get(variable, "sim"))
        return [axis_y, unit_z]
    elif units in ("si", "db"):
        unit_z = g_axis_si_units.get(category_full, g_axis_si_units.get(variable, "1"))
        scale = scale_factors.get(unit_z, 1.0)
        axis_z_data = unit_convert.rescale_list(axis_y, scale)

        if unit_z == "J/m3":
            unit_z = r"$J/m^{3}$"
        return [axis_z_data, unit_z]
    else:
        return [axis_y, f"Invalid unit type: {units}"]



def fft_data_transform(dataX, dataY):
    """Bezpečný výpočet 1D FFT bez pevných indexů."""
    if len(dataX) < 2 or len(dataY) < 2:
        return [[], []]

    step = float(np.diff(dataX).mean())
    if step == 0:
        return [[], []]

    fft_result = np.fft.fft(dataY)
    frequencies = np.fft.fftfreq(len(dataY), d=step)
    magnitude = np.abs(fft_result)

    n_half = len(frequencies) // 2
    return [frequencies[1:n_half], magnitude[1:n_half]]


def extract_unit_from_label(label: str) -> str:
    """Extrahuje fyzikální jednotku z popisku osy (uvnitř závorek [...])."""
    if "[" in label and "]" in label:
        _, _, unit_part = label.partition("[")
        unit1, _, _ = unit_part.partition("]")
        return unit1.strip()
    return label.strip()


def extract_half(data):
    return data[len(data) // 2:]