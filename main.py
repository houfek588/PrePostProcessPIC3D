"""
main.py - Hlavní vstupní skript pro postprocessing výsledků numerického řešiče iPIC3D.
"""

import os
import json
import post_process.vtk_processing as vtk
import post_process.hdf5_processing as hdf
import post_process.ploting as ploting
import post_process.data_processing as data
from post_process.config_manager import load_config, save_config_file, resolve_config_path

# Seznam podporovaných veličin pro GUI a skripty
g_available = [
    "Efield_x", "Efield_y", "Efield_z",
    "rhoe0", "rhoi1", "rhoe2",
    "velocity_1_x", "velocity_2_x", "velocity_3_x",
    "energy_kin", "energy_ele",
    "energy_kin0", "energy_kin1", "energy_kin2",
    "energy_total"
]


def check_saved_data(var: str):
    """
    Zkontroluje, zda již existují předpočtená data v mezipaměti script_data/.
    Pokud neexistují, načte je ze zdrojových simulačních souborů (VTK / HDF5 / ConservedQuantities).
    """
    script_data_folder = "script_data"
    os.makedirs(script_data_folder, exist_ok=True)
    files = os.listdir(script_data_folder)

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


def clear_script_data():
    """Vymaže dočasné .npy soubory z mezipaměti script_data/."""
    folder_path = "script_data"
    if not os.path.exists(folder_path):
        return

    count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            count += 1
    print(f"Promazána mezipaměť: smazáno {count} souborů ze složky '{folder_path}'.")


def data_process(single_var: list, multi_var: list):
    """
    Hlavní řídicí funkce pro zpracování a vykreslení vybraných veličin.
    """
    all_vars = list(dict.fromkeys(multi_var + single_var))
    print(f"\n=======================================================")
    print(f" Spouštím analýzu dat iPIC3D")
    print(f" Vybrané veličiny: {all_vars}")
    print(f"=======================================================\n")

    # 1. Kontrola a příprava dat v mezipaměti
    for var in all_vars:
        check_saved_data(var)

    # 2. Analýza jednotlivých veličin
    for s in single_var:
        data.single_result_analysis(s)

    # 3. Multi-analýza (srovnání více veličin)
    if multi_var:
        data.multi_result_analysis(multi_var)

    # 4. Zobrazení grafů na obrazovku (pokud je zapnuto v ploting)
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


if __name__ == '__main__':
    # Výchozí testovací spuštění:
    # multi_var = ["velocity_1_x", "velocity_3_x"]
    multi_var = []
    # single_var = ["Efield_x", "rhoe0", "energy_ele"]
    single_var = ["Efield_x", "energy_ele"]

    data_process(single_var, multi_var)
