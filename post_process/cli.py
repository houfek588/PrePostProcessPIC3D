"""
cli.py - Příkazová řádka (CLI) pro postprocessing simulací iPIC3D na serveru i lokálně.

Umožňuje přímé volání s parametry:
  python -m post_process -d /cesta/k/datum -o /cesta/pro/grafy -v Efield_x energy_ele

Nebo s načtením JSON konfigurace:
  python -m post_process -c config.json

Případně kombinací obojího (parametry z příkazové řádky mají přednost).
"""

import argparse
import os
import sys
from typing import List, Optional

try:
    from .core import data_process, clear_script_data, check_saved_data, AVAILABLE_VARIABLES, save_config, save_plot_config
    from .config_manager import (
        load_config, load_plot_config, set_active_config, update_active_config,
        set_active_plot_config, get_cache_dir, to_bool
    )
    from .plasma_units import PlasmaUnits, set_global_units
    from . import ploting
except ImportError:
    from core import data_process, clear_script_data, check_saved_data, AVAILABLE_VARIABLES, save_config, save_plot_config
    from config_manager import (
        load_config, load_plot_config, set_active_config, update_active_config,
        set_active_plot_config, get_cache_dir, to_bool
    )
    from plasma_units import PlasmaUnits, set_global_units
    import ploting

# Alias pro zpětnou kompatibilitu s původním main.py
g_available = AVAILABLE_VARIABLES


def build_parser() -> argparse.ArgumentParser:
    """Vytvoří a nakonfiguruje parser příkazové řádky."""
    parser = argparse.ArgumentParser(
        prog="post_process",
        description="Samostatný nástroj pro postprocessing a generování grafů z výsledků iPIC3D."
    )

    parser.add_argument("-d", "--data-dir", dest="data_dir", default=None,
                        help="Cesta ke složce se surovými daty simulace (VTK, HDF5, text).")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None,
                        help="Cesta ke složce pro uložení vygenerovaných grafů (.png).")
    parser.add_argument("-c", "--config", dest="config_file", default=None,
                        help="Cesta ke konfiguračnímu souboru JSON (např. config.json).")
    parser.add_argument("-p", "--plot-config", dest="plot_config_file", default=None,
                        help="Cesta ke konfiguračnímu souboru pro typy grafů (plot_config.json).")

    parser.add_argument("-v", "--vars", dest="single_vars", nargs="+", default=None,
                        help="Seznam veličin pro samostatnou analýzu (např. -v Efield_x energy_ele).")
    parser.add_argument("-m", "--multi-vars", dest="multi_vars", nargs="+", default=None,
                        help="Seznam veličin pro sdruženou analýzu (např. -m velocity_1_x velocity_3_x).")

    parser.add_argument("--plot-types", dest="plot_types", default=None,
                        help="Rychlé zapnutí typů grafů oddělených čárkou (val,fft,hist,val2d,fft2d).")

    parser.add_argument("--inp", dest="inp_file", default=None,
                        help="Cesta k .inp souboru simulace pro automatickou kalibraci fyzikálních jednotek a frekvencí.")

    parser.add_argument("--field-cycle", dest="field_cycle", type=int, default=None,
                        help="Krok ukládání polí v simulaci (FieldOutputCycle).")
    parser.add_argument("--particle-cycle", dest="particle_cycle", type=int, default=None,
                        help="Krok ukládání částic v simulaci (ParticleOutputCycle).")
    parser.add_argument("--num-hdf", dest="num_hdf", type=int, default=None,
                        help="Počet HDF5 souborů na časový krok (NumberHdfFiles).")

    parser.add_argument("--save-graphs", dest="save_graphs", action="store_true", default=None,
                        help="Ukládat vygenerované grafy na disk (výchozí).")
    parser.add_argument("--no-save", dest="save_graphs", action="store_false",
                        help="Neukládat grafy na disk.")

    parser.add_argument("--show", dest="show_plots", action="store_true", default=False,
                        help="Zobrazit grafy interaktivně na obrazovce (pro lokální použití s GUI).")

    parser.add_argument("--clean-cache", dest="clean_cache", action="store_true", default=False,
                        help="Před spuštěním promazat mezipaměť (script_data/).")
    parser.add_argument("--cache-dir", dest="cache_dir", default=None,
                        help="Vlastní složka pro dočasné soubory mezipaměti.")

    parser.add_argument("--list-vars", dest="list_vars", action="store_true", default=False,
                        help="Vypíše všechny podporované veličiny a ukončí program.")

    return parser


def parse_plot_types(plot_types_str: str) -> dict:
    """Převede řetězec jako 'val,fft,hist,fft2d' na strukturovaný konfigurační slovník."""
    types = [t.strip().lower() for t in plot_types_str.split(",")]
    return {
        "Value": str("val" in types or "value" in types),
        "FFT": str("fft" in types),
        "Histogram": str("hist" in types or "histogram" in types),
        "Value2D": str("val2d" in types or "value2d" in types or "2d" in types),
        "FFT2D": str("fft2d" in types),
        "FFT in time": str("ffttime" in types or "timefft" in types)
    }


def main(args_list: Optional[List[str]] = None) -> int:
    """Hlavní vstupní funkce pro CLI."""
    parser = build_parser()
    args = parser.parse_args(args_list)

    # 1. Výpis podporovaných veličin
    if args.list_vars:
        print("\nPodporované veličiny pro iPIC3D postprocessing:")
        print("------------------------------------------------")
        for v in AVAILABLE_VARIABLES:
            print(f"  - {v}")
        print()
        return 0

    # 2. Nastavení grafického backendu pro headless servery
    if not args.show_plots:
        ploting.setup_backend(headless=True)

    # 3. Načtení základní konfigurace z JSON (pokud existuje)
    if args.config_file:
        base_cfg = load_config(args.config_file, reload=True)
    else:
        base_cfg = load_config(reload=True)

    # 4. Aplikace CLI argumentů s vyšší prioritou
    overrides = {}
    if args.data_dir is not None:
        overrides["result_folder"] = args.data_dir
    if args.output_dir is not None:
        overrides["output_folder"] = args.output_dir
    if args.cache_dir is not None:
        overrides["cache_dir"] = args.cache_dir
    if args.save_graphs is not None:
        overrides["save_graphs"] = str(args.save_graphs)
    if args.field_cycle is not None:
        overrides["FieldOutputCycle"] = args.field_cycle
    if args.particle_cycle is not None:
        overrides["ParticleOutputCycle"] = args.particle_cycle
    if args.num_hdf is not None:
        overrides["NumberHdfFiles"] = args.num_hdf

    # Aktualizace aktivní konfigurace
    if overrides:
        update_active_config(overrides)

    # Zajištění platných cest
    active_cfg = load_config()
    res_folder = active_cfg.get("result_folder", "./")
    out_folder = active_cfg.get("output_folder", "./result")
    os.makedirs(out_folder, exist_ok=True)

    # 5. Konfigurace typů grafů
    if args.plot_config_file:
        load_plot_config(args.plot_config_file, reload=True)

    if args.plot_types:
        parsed_single = parse_plot_types(args.plot_types)
        plot_cfg = load_plot_config()
        plot_cfg["single"] = parsed_single
        set_active_plot_config(plot_cfg)

    # 6. Kalibrace fyzikálních jednotek ze souboru .inp (pokud zadán)
    if args.inp_file:
        print(f">> Kalibruji fyzikální jednotky z .inp souboru: {args.inp_file}")
        units = PlasmaUnits.from_inp_file(args.inp_file)
        set_global_units(units)
        print("  Fyzikální jednotky úspěšně nastaveny.")

    # 7. Čištění mezipaměti
    cache_path = get_cache_dir()
    if args.clean_cache:
        clear_script_data(cache_path)

    # 8. Určení veličin k analýze
    single_vars = args.single_vars
    multi_vars = args.multi_vars if args.multi_vars else []

    if single_vars is None:
        # Výchozí sada veličin pro analýzu, pokud nebylo zadáno
        single_vars = ["Efield_x", "energy_ele"]
        print(f"[Info] Nebyly zadány žádné veličiny přepínačem -v. Používám výchozí: {single_vars}")

    # 9. Spuštění analýzy
    try:
        data_process(
            single_var=single_vars,
            multi_var=multi_vars,
            cache_dir=cache_path,
            show_plots=args.show_plots
        )
    except Exception as e:
        print(f"\n[CHYBA] Běh postprocessingu selhal: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
