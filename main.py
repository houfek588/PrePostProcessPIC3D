"""
main.py - Hlavní vstupní skript pro postprocessing výsledků numerického řešiče iPIC3D.

Funguje jako zpětně kompatibilní wrapper nad samostatným balíčkem post_process.
Lze jej volat z terminálu s parametry i z grafického rozhraní (win.py).
"""

import sys
from post_process.cli import (
    main,
    data_process,
    clear_script_data,
    check_saved_data,
    save_config,
    save_plot_config,
    g_available,
    AVAILABLE_VARIABLES
)

if __name__ == "__main__":
    sys.exit(main())
