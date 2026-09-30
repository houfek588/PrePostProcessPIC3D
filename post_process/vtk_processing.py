"""
vtk_processing.py - Zpracování výstupních VTK souborů z iPIC3D do NumPy mezipaměti.
"""

import os
import re
import numpy as np

try:
    from . import read_files as read
    from .config_manager import load_config, to_bool, get_cache_dir
except ImportError:
    import read_files as read
    from config_manager import load_config, to_bool, get_cache_dir



def extract_number(filename: str):
    """Extrahuje číslo z názvu souboru pro správné numerické řazení."""
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    return int(match.group()) if match else float("inf")


def process_folder(folder: str, variable_name: str, sort_key_func=extract_number) -> list:
    """Filtruje a seřadí .vtk soubory v dané složce podle názvu proměnné."""
    if not os.path.exists(folder):
        print(f"[Warning] Složka '{folder}' neexistuje.")
        return []

    files = os.listdir(folder)
    if not files:
        print(f"[Warning] Složka '{folder}' je prázdná.")
        return []

    vtk_files = [f for f in files if f.endswith(".vtk")]
    par_files = [f for f in vtk_files if variable_name in f]
    par_sorted_files = sorted(par_files, key=sort_key_func)

    print(f"Zdrojová složka má {len(par_sorted_files)} VTK souborů k analýze pro '{variable_name}'...")
    return par_sorted_files


def read_step_size(data_type: str = "field") -> int:
    """Vrátí krok ukládání polí nebo částic z konfigurace."""
    cfg = load_config()
    if data_type == "part":
        return int(cfg.get("ParticleOutputCycle", 100000))
    return int(cfg.get("FieldOutputCycle", 10))


def result_analysis(set_data: str):
    """
    Analyzuje VTK soubory pro zadanou proměnnou (např. 'Efield_x', 'rhoe0')
    a uloží 2D NumPy pole do script_data/{set_data}_data2D.npy.
    """
    print(f"Spouštím VTK analýzu pro: {set_data}...")

    cfg = load_config()
    folder = cfg.get("result_folder", "")
    step = read_step_size("field")

    # Načtení simulačních metadat z settings.hdf (pokud existuje)
    settings_path = os.path.join(folder, "settings.hdf")
    setting = read.ReadHDFSettings(settings_path)
    nx = setting.get_num_cells("x")
    nz = setting.get_num_cells("z")
    dt = setting.get_time_step_size()
    num_of_files = setting.get_time_step_cycles()

    # Rozdělení názvu na kategorii a osu (např. 'Efield_x' -> 'Efield', 'x')
    name_split = set_data.split("_")
    category = name_split[0] if name_split else ""
    axis = name_split[1] if len(name_split) > 1 else "x"

    print(f"Veličina k analýze: {category}, složka: {axis}")

    field_map = {"Efield": "E", "Bfield": "B"}
    special_cases = {"rhoe", "rhoi", "Je", "Ji"}

    if category in field_map:
        variable_name = field_map[category]
    elif (category + axis) in special_cases:
        variable_name = category + axis
    else:
        variable_name = category

    par_sorted_files = process_folder(folder, f"_{variable_name}_", extract_number)
    if not par_sorted_files:
        # Zkusíme bez podtržítek
        par_sorted_files = process_folder(folder, str(variable_name), extract_number)

    if not par_sorted_files:
        raise FileNotFoundError(f"Nebyly nalezeny žádné VTK soubory pro proměnnou '{variable_name}' v '{folder}'.")

    pic_data = read.ReadVTKFilesData(folder, par_sorted_files, num_of_files, step, dt, nx, nz, variable_name, axis)

    proc_var_dict = {
        "Efield": "Efield",
        "Bfield": "Bfield",
        "rhoe0": "rhoe0",
        "rhoi1": "rhoi1",
        "rhoe2": "rhoe2",
        "Je": "electron_current",
        "Ji": "ion_current",
    }

    base_cat = proc_var_dict.get(category, category)
    if pic_data.is_vector():
        out_var_name = f"{base_cat}_{axis}"
    else:
        out_var_name = f"{base_cat}"

    # Zajištění existence složky mezipaměti
    cache_dir = get_cache_dir()
    output_file = os.path.join(cache_dir, f"{out_var_name}_data2D.npy")

    np.save(output_file, pic_data.get_2D_data())
    print(f"VTK data úspěšně uložena do mezipaměti: {output_file}")


def conserve_analysis(set_data: str):
    """
    Analyzuje integrální konzervované veličiny (energie) ze souboru ConservedQuantities.txt
    a uloží je jako 1D NumPy pole do {cache_dir}/{set_data}_data1D.npy.
    """
    cfg = load_config()
    folder = cfg.get("result_folder", "")

    energy_data = read.ReadConsData(folder, "ConservedQuantities.txt")

    supported = {
        "energy_kin": energy_data.get_k_energy,
        "energy_ele": energy_data.get_e_energy,
        "energy_kin0": energy_data.get_k_spec0_energy,
        "energy_kin1": energy_data.get_k_spec1_energy,
        "energy_kin2": energy_data.get_k_spec2_energy,
        "energy_total": energy_data.get_total_energy,
    }

    if set_data not in supported:
        raise ValueError(f"Nepodporovaná veličina energií: '{set_data}'. Dostupné: {list(supported.keys())}")

    cache_dir = get_cache_dir()
    new_file_name = os.path.join(cache_dir, f"{set_data}_data1D.npy")

    data = supported[set_data]()  # Vrací tuple: (title, list_of_values)
    values = data[1] if isinstance(data, (list, tuple)) and len(data) > 1 else data
    np.save(new_file_name, np.asarray(values, dtype=float))
    print(f"Data energií uložena do mezipaměti: {new_file_name}")

