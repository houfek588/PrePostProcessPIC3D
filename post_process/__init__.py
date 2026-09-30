"""
post_process - Samostatný balíček pro analýzu a vizualizaci dat numerického řešiče iPIC3D.
"""

import sys
import os

# Zajištění možnosti importu modulů balíčku v libovolném pracovním adresáři
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from .core import (
    data_process,
    clear_script_data,
    check_saved_data,
    AVAILABLE_VARIABLES,
    save_config,
    save_plot_config
)
from .config_manager import (
    load_config,
    load_plot_config,
    set_active_config,
    get_active_config,
    update_active_config,
    get_cache_dir,
    to_bool
)
from .plasma_units import (
    PlasmaUnits,
    get_default_units,
    set_global_units
)
from .cli import main

__all__ = [
    "data_process",
    "clear_script_data",
    "check_saved_data",
    "AVAILABLE_VARIABLES",
    "save_config",
    "save_plot_config",
    "load_config",
    "load_plot_config",
    "set_active_config",
    "get_active_config",
    "update_active_config",
    "get_cache_dir",
    "to_bool",
    "PlasmaUnits",
    "get_default_units",
    "set_global_units",
    "main",
]
