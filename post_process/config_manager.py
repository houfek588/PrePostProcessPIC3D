"""
config_manager.py - Centralizovaná správa konfigurací pro postprocessing.

Zajišťuje:
- Bezpečné načítání konfigurací (s vyhledáním v cwd, v balíčku i v kořenu projektu).
- Podporu pro CLI overrides a in-memory konfiguraci bez nutnosti existence souborů na disku.
- Dynamickou správu cesty k mezipaměti script_data/.
- Kompatibilitu s Python 3.12+ (vlastní funkce to_bool).
"""

import json
import os
from typing import Any, Dict, Optional

# Výchozí konfigurace pro případ, kdy není dodán žádný konfigurační soubor
DEFAULT_CONFIG: Dict[str, Any] = {
    "result_folder": "./",
    "output_folder": "./result",
    "cache_dir": "script_data",
    "save_graphs": "True",
    "FieldOutputCycle": 10,
    "ParticleOutputCycle": 100000,
    "NumberHdfFiles": 32
}

DEFAULT_PLOT_CONFIG: Dict[str, Any] = {
    "single": {
        "Value": "True",
        "FFT": "False",
        "Histogram": "False",
        "Value2D": "False",
        "FFT2D": "True",
        "FFT in time": "False"
    },
    "multi": {
        "Value": "False",
        "FFT": "False",
        "Histogram": "True"
    }
}

_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}
_ACTIVE_CONFIG: Optional[Dict[str, Any]] = None
_ACTIVE_PLOT_CONFIG: Optional[Dict[str, Any]] = None


def to_bool(val: Any) -> bool:
    """
    Převede vstupní hodnotu na boolean.
    Nahrazuje distutils.util.strtobool pro plnou kompatibilitu s Python 3.12+.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes", "t", "on")


def get_package_dir() -> str:
    """Vrátí absolutní cestu ke složce post_process."""
    return os.path.dirname(os.path.abspath(__file__))


def get_project_root() -> str:
    """Vrátí absolutní cestu k nadřazené složce balíčku (PrePostProcessPIC3D)."""
    return os.path.dirname(get_package_dir())


def resolve_config_path(file_name: str) -> Optional[str]:
    """
    Vyřeší cestu ke konfiguračnímu souboru. Hledá v tomto pořadí:
    1. Absolutní či relativní cesta v aktuálním pracovním adresáři (cwd).
    2. Ve složce balíčku post_process.
    3. V nadřazeném projektovém adresáři.
    """
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name
    if os.path.exists(file_name):
        return os.path.abspath(file_name)

    # Hledání ve složce balíčku
    pkg_path = os.path.join(get_package_dir(), file_name)
    if os.path.exists(pkg_path):
        return pkg_path

    # Hledání v nadřazené složce
    root_path = os.path.join(get_project_root(), file_name)
    if os.path.exists(root_path):
        return root_path

    return None


def get_active_config() -> Dict[str, Any]:
    """Vrátí aktuální aktivní konfiguraci (nebo výchozí, pokud není nastavena)."""
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is None:
        _ACTIVE_CONFIG = dict(DEFAULT_CONFIG)
    return _ACTIVE_CONFIG


def set_active_config(cfg: Dict[str, Any]) -> None:
    """Nastaví aktivní konfiguraci (např. z CLI parametrů)."""
    global _ACTIVE_CONFIG
    base = dict(DEFAULT_CONFIG)
    base.update(cfg)
    _ACTIVE_CONFIG = base


def update_active_config(overrides: Dict[str, Any]) -> None:
    """Aktualizuje existující aktivní konfiguraci o předané hodnoty."""
    cfg = get_active_config()
    cfg.update(overrides)


def get_active_plot_config() -> Dict[str, Any]:
    """Vrátí aktuální konfiguraci vykreslování."""
    global _ACTIVE_PLOT_CONFIG
    if _ACTIVE_PLOT_CONFIG is None:
        _ACTIVE_PLOT_CONFIG = json.loads(json.dumps(DEFAULT_PLOT_CONFIG))
    return _ACTIVE_PLOT_CONFIG


def set_active_plot_config(plot_cfg: Dict[str, Any]) -> None:
    """Nastaví aktivní konfiguraci grafů."""
    global _ACTIVE_PLOT_CONFIG
    _ACTIVE_PLOT_CONFIG = plot_cfg


def get_cache_dir() -> str:
    """Vrátí cestu ke složce pro mezipaměť .npy souborů."""
    cfg = get_active_config()
    cache_path = cfg.get("cache_dir", "script_data")
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def load_config(file_name: str = "config.json", reload: bool = False) -> Dict[str, Any]:
    """
    Načte konfigurační soubor. Pokud byla nastavena aktivní in-memory konfigurace
    a soubor nebyl explicitně změněn, vrátí ji. Pokud soubor neexistuje,
    vrátí výchozí konfiguraci.
    """
    global _ACTIVE_CONFIG
    path = resolve_config_path(file_name)

    if path and not reload and path in _CONFIG_CACHE:
        return _CONFIG_CACHE[path]

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _CONFIG_CACHE[path] = data
        if _ACTIVE_CONFIG is None:
            _ACTIVE_CONFIG = dict(DEFAULT_CONFIG)
            _ACTIVE_CONFIG.update(data)
        return data

    # Soubor nebyl nalezen - pokud již máme aktivní konfiguraci, použijeme ji, jinak default
    return get_active_config()


def load_plot_config(file_name: str = "plot_config.json", reload: bool = False) -> Dict[str, Any]:
    """
    Načte konfiguraci vykreslování. Pokud soubor neexistuje, vrátí výchozí šablonu.
    """
    global _ACTIVE_PLOT_CONFIG
    path = resolve_config_path(file_name)

    if path and not reload and path in _CONFIG_CACHE:
        return _CONFIG_CACHE[path]

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _CONFIG_CACHE[path] = data
        if _ACTIVE_PLOT_CONFIG is None:
            _ACTIVE_PLOT_CONFIG = data
        return data

    return get_active_plot_config()


def save_config_file(data: Dict[str, Any], file_name: str = "config.json") -> str:
    """
    Uloží data do konfiguračního souboru v kořenu projektu a zneplatní cache.
    """
    path = resolve_config_path(file_name)
    if not path:
        path = os.path.join(get_project_root(), file_name)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    _CONFIG_CACHE[path] = data
    set_active_config(data)
    return path
