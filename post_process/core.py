"""
core.py - Jádro zpracování dat pro iPIC3D postprocessing.

Obsahuje orchestraci celého procesu:
- Kontrola mezipaměti (script_data/).
- Načtení chybějících dat ze zdrojových simulačních souborů (VTK, HDF5, text).
- Spuštění 1D/2D analýz a vykreslení grafů.
- Správa mezipaměti.
"""

import os
import json
from typing import List, Optional, Dict, Any

try:
    from . import vtk_processing as vtk
    from . import hdf5_processing as hdf
    from . import ploting
    from . import data_processing as data
    from .config_manager import load_config, save_config_file, get_cache_dir, resolve_config_path
except ImportError:
    import vtk_processing as vtk
    import hdf5_processing as hdf
    import ploting
    import data_processing as data
    from config_manager import load_config, save_config_file, get_cache_dir, resolve_config_path

# Seznam všech podporovaných veličin pro CLI, GUI a skripty
AVAILABLE_VARIABLES = [
    "Efield_x", "Efield_y", "Efield_z",
    "rhoe0", "rhoi1", "rhoe2",
    "velocity_1_x", "velocity_2_x", "velocity_3_x",
    "energy_kin", "energy_ele",
    "energy_kin0", "energy_kin1", "energy_kin2",
    "energy_total"
]


def check_saved_data(var: str, cache_dir: Optional[str] = None):
    """
    Zkontroluje, zda již existují předpočtená data v mezipaměti.
    Pokud neexistují, načte je ze zdrojových simulačních souborů (VTK / HDF5 / ConservedQuantities).
    """
    target_cache = cache_dir if cache_dir else get_cache_dir()
    os.makedirs(target_cache, exist_ok=True)
    files = os.listdir(target_cache)

    # Hledáme soubory tvaru {var}_data2D.npy nebo {var}_data1D.npy
    found = any(f.startswith(var) and f.endswith(".npy") for f in files)

    status_str = "Nalezena v mezipaměti (rychlé načtení)" if found else "Nenalezena v mezipaměti, načítám ze surových dat..."
    print(f">> Veličina '{var}': {status_str}")

    if not found:
        # 1. Energie (uloženy v textovém souboru ConservedQuantities.txt)
        if var.startswith("energy_") or var == "energy":
            print(f"  Načítám data energií z ConservedQuantities.txt pro '{var}'...")
            try:
                vtk.conserve_analysis(var)
                print(f"  Data energií pro '{var}' úspěšně načtena.")
                return
            except Exception as e:
                raise RuntimeError(f"Chyba při načítání energií pro '{var}': {e}")

        # 2. Částicová data (rychlosti/pozice druhů uložené v HDF5)
        elif var.startswith("velocity") or var.startswith("position"):
            print(f"  Načítám částicová data z HDF5 pro '{var}'...")
            try:
                hdf.result_analysis(var)
                print(f"  Částicová data pro '{var}' úspěšně načtena.")
                return
            except Exception as e:
                raise RuntimeError(f"Chyba při načítání částic z HDF5 pro '{var}': {e}")

        # 3. Polní veličiny (Efield, Bfield, rhoe, rhoi, Je, Ji)
        else:
            print(f"  Načítám polní data pro '{var}'...")
            vtk_err = None
            try:
                vtk.result_analysis(var)
                print(f"  VTK data pro '{var}' úspěšně načtena.")
                return
            except Exception as e:
                vtk_err = e

            # Fallback na HDF5 pokud VTK není k dispozici
            try:
                hdf.result_analysis(var)
                print(f"  HDF5 data pro '{var}' úspěšně načtena (fallback).")
                return
            except Exception as hdf_err:
                raise RuntimeError(f"Veličinu '{var}' se nepodařilo načíst ani z VTK ({vtk_err}), ani z HDF5 ({hdf_err}).")


def clear_script_data(cache_dir: Optional[str] = None):
    """Vymaže dočasné .npy soubory z mezipaměti."""
    target_cache = cache_dir if cache_dir else get_cache_dir()
    if not os.path.exists(target_cache):
        return

    count = 0
    for filename in os.listdir(target_cache):
        file_path = os.path.join(target_cache, filename)
        if os.path.isfile(file_path) and filename.endswith(".npy"):
            os.remove(file_path)
            count += 1
    print(f"Promazána mezipaměť: smazáno {count} .npy souborů ze složky '{target_cache}'.")


def data_process(single_var: List[str], multi_var: Optional[List[str]] = None,
                 cache_dir: Optional[str] = None, show_plots: bool = False):
    """
    Hlavní řídicí funkce pro zpracování a vykreslení vybraných veličin.
    """
    if multi_var is None:
        multi_var = []

    all_vars = list(dict.fromkeys(multi_var + single_var))
    print(f"\n=======================================================")
    print(f" Spouštím analýzu dat iPIC3D")
    print(f" Vybrané veličiny: {all_vars}")
    print(f"=======================================================\n")

    # 1. Kontrola a příprava dat v mezipaměti
    for var in all_vars:
        check_saved_data(var, cache_dir=cache_dir)

    # 2. Analýza jednotlivých veličin
    for s in single_var:
        data.single_result_analysis(s)

    # 3. Multi-analýza (srovnání více veličin)
    if multi_var:
        data.multi_result_analysis(multi_var)

    # 4. Zobrazení grafů na obrazovku (pokud je explicitně vyžádáno)
    if show_plots:
        ploting.plot_all_graphs()

    print("\n[OK] Analýza iPIC3D dat úspěšně dokončena.")


def save_config(result_folder: str, output_folder: str, save_graphs: bool,
                field_output_cycle: int, particle_output_cycle: int, number_hdf_files: int):
    """Uloží hlavní konfiguraci do config.json."""
    config_dict = {
        "result_folder": result_folder,
        "output_folder": output_folder,
        "save_graphs": str(save_graphs),
        "FieldOutputCycle": field_output_cycle,
        "ParticleOutputCycle": particle_output_cycle,
        "NumberHdfFiles": number_hdf_files
    }
    path = save_config_file(config_dict, "config.json")
    print(f"Konfigurace úspěšně uložena do: {path}")


def save_plot_config(options_single: dict, options_multi: dict):
    """Uloží konfiguraci vykreslování do plot_config.json."""
    plot_dict = {
        "single": options_single,
        "multi": options_multi
    }
    path = save_config_file(plot_dict, "plot_config.json")
    print(f"Konfigurace grafů úspěšně uložena do: {path}")
