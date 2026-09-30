"""
hdf5_processing.py - Zpracování částicových a polních dat z HDF5 souborů iPIC3D do mezipaměti.
"""

import os
import numpy as np

try:
    from . import read_files as read
    from . import ploting
    from . import unit_convert
    from .config_manager import load_config, to_bool, get_cache_dir
    from .plasma_units import get_default_units
except ImportError:
    import read_files as read
    import ploting
    import unit_convert
    from config_manager import load_config, to_bool, get_cache_dir
    from plasma_units import get_default_units



def create_particle_dict(ns: int) -> dict:
    """Vytvoří mapování pro částicové druhy (velocity a position podle složek x, y, z)."""
    particle_dict = {"velocity": {}, "position": {}}
    for i in range(1, ns + 1):
        species = f"species_{i - 1}"
        particle_dict["velocity"][str(i)] = {"x": [species, "u"], "y": [species, "v"], "z": [species, "w"]}
        particle_dict["position"][str(i)] = {"x": [species, "x"], "y": [species, "y"], "z": [species, "z"]}
    return particle_dict


def result_analysis(set_data: str):
    """
    Analyzuje částicová data z HDF5 souborů a uloží výsledek jako 2D NumPy pole
    do script_data/{set_data}_data2D.npy.
    """
    print(f"Spouštím HDF5 analýzu pro: {set_data}...")

    cfg = load_config()
    folder = cfg.get("result_folder", "")
    number_files = int(cfg.get("NumberHdfFiles", 32))
    hdf_file = "proc0.hdf"

    settings_path = os.path.join(folder, "settings.hdf")
    setting = read.ReadHDFSettings(settings_path)
    ns = setting.get_num_species()

    particle_dict = create_particle_dict(ns)

    try:
        category, num, axis = set_data.split("_")
    except ValueError:
        raise ValueError(f"Neplatný formát pro set_data: '{set_data}'. Očekává se formát 'category_num_axis' (např. 'velocity_1_x').")

    print(f"Kategorie: {category}, Částicový druh: {num}, Osa: {axis}")

    try:
        species, symbol = particle_dict[category][num][axis]
    except KeyError:
        raise KeyError(f"Nenalezen odpovídající druh/symbol pro: '{set_data}' v počtu druhů ns={ns}.")

    print(f"Extrahuji data pro druh: {species}, symbol: {symbol}")

    p_file = read.ReadHDFParticleData(folder, hdf_file, number_files, species, symbol)

    cache_dir = get_cache_dir()
    new_file_name = os.path.join(cache_dir, f"{set_data}_data2D.npy")
    np.save(new_file_name, p_file.data_x_t)

    print(f"HDF5 data úspěšně uložena do mezipaměti: {new_file_name}")



def get_axis_data(data, setting):
    """Přepočte osy délky a času do jednotek simulace (pic), SI a Debyeových délek (db)."""
    unit = get_default_units()
    axis_x = data.get_len_data()
    axis_t = data.get_time_data()

    box_size = setting.get_box_size("x")
    num_cells = setting.get_num_cells("x")
    dx_factor = (unit.ion.get_ion_skin_depth() * box_size / num_cells) if num_cells > 0 else 1.0

    axis_x_si = unit_convert.rescale_list(axis_x, dx_factor)
    axis_time_si = unit_convert.rescale_list(axis_t, 1.0 / unit.ion.get_plasma_frequency())

    axis_x_dict = {
        "pic": axis_x,
        "SI": axis_x_si,
        "db": unit_convert.rescale_list(axis_x_si, 1.0 / unit.debye_len),
    }

    axis_t_dict = {
        "pic": axis_t,
        "SI": axis_time_si,
        "om": unit_convert.rescale_list(axis_time_si, unit.electron.get_plasma_frequency()),
    }

    return [axis_x_dict, axis_t_dict]


def particle_graphs(data: read.ReadHDFParticleData, par: dict, sett):
    """Vykreslí grafy pro částicová data."""
    unit = get_default_units()
    axis_x_dict, axis_t_dict = get_axis_data(data, sett)
    units = "SI"
    data_name = data.get_data_name()

    save = to_bool(par.get("save_graphs", True))
    save_file_path = par.get("output_folder", "")

    t = int(par.get("visualization_parameters", {}).get("plot_length", {}).get("time_parameter", -1))
    vis_plot_len = par.get("visualization_parameters", {}).get("plot_length", {})
    save_file_name = (
        par.get("particle", {}).get("axis", "x")
        + "_"
        + par.get("particle", {}).get("type", "part")
        + "_"
        + vis_plot_len.get("file_name", "plot.png")
    )
    enable_fft = to_bool(vis_plot_len.get("enable_fft", False))
    enable_histogram = True

    velocity = data.get_particle1D_len(t)
    velocity_si = unit_convert.rescale_list(velocity, unit.const_c)


    description = ploting.PlotDescription(
        f"Length {data_name} data for t({round(axis_t_dict[units][t], 3)})",
        "length [m]",
        "velocity [m/s]",
    )
    ploting.plot_data(axis_x_dict[units], velocity_si, description, save, os.path.join(save_file_path, save_file_name))

    if enable_histogram:
        save_file_name_hist = "hist_" + save_file_name
        description_hist = ploting.PlotDescription(
            f"Histogram for {data_name}; t = {round(axis_t_dict[units][t], 3)} s",
            f"{par.get('particle', {}).get('axis', 'x')} [m/s]",
            "Magnitude",
        )
        ploting.plot_histogram(velocity, 4096, description_hist, [1.0] * len(velocity), save, os.path.join(save_file_path, save_file_name_hist))

    if enable_fft:
        save_file_name_fft = "fft_" + save_file_name
        description_fft = ploting.PlotDescription(
            f"Frequency Spectrum for {data_name}; t = {round(axis_t_dict[units][t], 3)} s",
            "Wavenumber [m-1]",
            "Magnitude",
        )
        ploting.plot_fft(axis_x_dict["SI"], velocity_si, description_fft, save, os.path.join(save_file_path, save_file_name_fft))


def field_graphs(data: read.ReadHDFFieldData, par: dict, sett):
    """Vykreslí 2D časoprostorový vývoj polí z HDF."""
    axis_x_dict, axis_t_dict = get_axis_data(data, sett)
    units = "SI"

    save = to_bool(par.get("save_graphs", True))
    save_file_path = par.get("output_folder", "")
    vis_2d = par.get("visualization_parameters", {}).get("plot_2D", {})
    save_file_name = par.get("field", {}).get("variable", "field") + "_" + vis_2d.get("file_name", "2d_plot.png")

    data_z = data.get_2D_data()
    description3d = ploting.PlotDescription(
        f"Time development through space for {data.get_data_name()}",
        "length [db]",
        "time [1/Om_pi]",
        "E",
    )
    ploting.plot3Dplane_data(axis_x_dict[units], axis_t_dict[units], data_z, description3d, save, os.path.join(save_file_path, save_file_name))