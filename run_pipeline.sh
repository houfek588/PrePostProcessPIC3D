#!/bin/bash
# ==============================================================================
# run_pipeline.sh - Automatizovaný běh simulace iPIC3D a navazujícího postprocessingu
# ==============================================================================
# Použití:
#   ./run_pipeline.sh <nazev_inp_souboru> [pocet_jader]
#
# Příklady:
#   ./run_pipeline.sh beam_3pop_drift_B.inp
#   ./run_pipeline.sh beam_3pop_drift_B.inp 32
#   ./run_pipeline.sh inputfiles/beam_3pop_drift_B.inp 64
# ==============================================================================

set -euo pipefail

# Barevný výstup pro přehlednost v konzoli
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ------------------------------------------------------------------------------
# 1. URČENÍ PRACOVNÍCH SLOŽEK (VÝPOČET Z iPIC3D/build)
# ------------------------------------------------------------------------------
START_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Automatická lokalizace složky iPIC3D/build:
if [ -f "./iPIC3D" ] && [ -d "./inputfiles" ]; then
    # Skript byl spuštěn přímo ze složky iPIC3D/build
    BUILD_DIR="$(pwd)"
elif [ -d "./iPIC3D/build" ]; then
    # Skript byl spuštěn z nadřazené složky (např. PIC_3D)
    BUILD_DIR="$(cd "./iPIC3D/build" && pwd)"
elif [ -d "${SCRIPT_DIR}/../iPIC3D/build" ]; then
    BUILD_DIR="$(cd "${SCRIPT_DIR}/../iPIC3D/build" && pwd)"
elif [ -d "${HOME}/iPIC3D/build" ]; then
    BUILD_DIR="${HOME}/iPIC3D/build"
else
    BUILD_DIR="$(pwd)/iPIC3D/build"
fi

DATA_DIR="${DATA_DIR:-${BUILD_DIR}/data}"
INPUT_DIR="${INPUT_DIR:-${BUILD_DIR}/inputfiles}"
RESULTS_DIR="${RESULTS_DIR:-${START_DIR}/results}"
LOG_DIR="${LOG_DIR:-${START_DIR}/logs}"

# Nastavení výpočtu iPIC3D
MPI_EXEC="${MPI_EXEC:-/lib64/mpich/bin/mpiexec}"
NUM_PROCS="${2:-${NUM_PROCS:-32}}"

# Nastavení Pythonu a virtuálního prostředí
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-${START_DIR}/venv}"

# ------------------------------------------------------------------------------
# KROK 1: Kontrola existence INP souboru v iPIC3D/build/inputfiles/
# ------------------------------------------------------------------------------
INP_INPUT="${1:-}"

if [ -z "$INP_INPUT" ]; then
    log_error "Nebyl zadán žádný .inp soubor!"
    echo -e "Použití:\n  $0 <nazev_inp_souboru> [pocet_jader]\n"
    echo -e "Příklady:\n  $0 beam_3pop_drift_B.inp\n  $0 beam_3pop_drift_B.inp 32"
    exit 1
fi

INP_FILENAME="$(basename "$INP_INPUT")"
INP_FULL_PATH="${INPUT_DIR}/${INP_FILENAME}"

if [ ! -f "$INP_FULL_PATH" ]; then
    # Pokud soubor není v INPUT_DIR, zkusíme, zda nebyla zadána přímá cesta k existujícímu souboru
    if [ -f "$INP_INPUT" ]; then
        mkdir -p "$INPUT_DIR"
        log_info "Kopíruji zadaný soubor '$INP_INPUT' do '${INPUT_DIR}/'..."
        cp "$INP_INPUT" "$INP_FULL_PATH"
    else
        log_error "Vstupní soubor '$INP_FILENAME' nebyl nalezen v '${INPUT_DIR}/'!"
        exit 1
    fi
fi

log_success "Nalezen vstupní soubor: ${INPUT_DIR}/${INP_FILENAME}"

# ------------------------------------------------------------------------------
# KROK 2: Kontrola a promazání složky iPIC3D/build/data/
# ------------------------------------------------------------------------------
mkdir -p "$DATA_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

log_info "Kontroluji obsah datové složky: $DATA_DIR"

# Bezpečnostní pojistka proti smazání nesprávného adresáře
REAL_DATA_DIR="$(realpath "$DATA_DIR" 2>/dev/null || readlink -f "$DATA_DIR" 2>/dev/null || echo "$DATA_DIR")"
if [ "$REAL_DATA_DIR" = "/" ] || [ "$REAL_DATA_DIR" = "/home" ] || [ "$REAL_DATA_DIR" = "$HOME" ]; then
    log_error "Kritická bezpečnostní chyba: DATA_DIR ukazuje na '$REAL_DATA_DIR'! Zastavuji skript."
    exit 1
fi

ITEM_COUNT=$(find "$DATA_DIR" -mindepth 1 | wc -l)

if [ "$ITEM_COUNT" -gt 0 ]; then
    log_warning "Složka '$DATA_DIR' obsahuje $ITEM_COUNT souborů z předchozí simulace."
    log_info "Mažu obsah složky '$DATA_DIR'..."
    find "$DATA_DIR" -mindepth 1 -delete
    log_success "Složka '$DATA_DIR' byla vyčištěna."
else
    log_info "Složka '$DATA_DIR' je prázdná, čištění není nutné."
fi

# ------------------------------------------------------------------------------
# KROK 3 a 4: Spuštění výpočtu ZE SLOŽKY iPIC3D/build A ČEKÁNÍ NA DOKONČENÍ
# ------------------------------------------------------------------------------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SIM_LOG="${LOG_DIR}/sim_${TIMESTAMP}.log"

if [ ! -d "$BUILD_DIR" ]; then
    log_error "Složka '$BUILD_DIR' neexistuje!"
    exit 1
fi

if [ ! -f "${BUILD_DIR}/iPIC3D" ]; then
    log_error "Binárka iPIC3D nebyla nalezena v '${BUILD_DIR}/iPIC3D'!"
    exit 1
fi

if [ ! -x "${BUILD_DIR}/iPIC3D" ]; then
    log_warning "Nastavuji spustitelný příznak chmod +x na '${BUILD_DIR}/iPIC3D'..."
    chmod +x "${BUILD_DIR}/iPIC3D"
fi

if ! command -v "$MPI_EXEC" &> /dev/null && [ ! -x "$MPI_EXEC" ]; then
    log_error "Spouštěč MPI '$MPI_EXEC' nebyl nalezen! Zkontrolujte cestu k mpiexec/mpirun."
    exit 1
fi

# PŘEPNUTÍ DO SLOŽKY iPIC3D/build PŘED SPUŠTĚNÍM VÝPOČTU
log_info "Přepínám pracovní adresář do: $BUILD_DIR"
cd "$BUILD_DIR"

SPUSTECI_PRIKAZ="$MPI_EXEC -n $NUM_PROCS ./iPIC3D inputfiles/$INP_FILENAME"

echo ""
log_info "================================================================="
log_info " SPOUŠTÍM SIMULACI iPIC3D"
log_info " Pracovní složka : $(pwd)"
log_info " Spouštěcí příkaz: $SPUSTECI_PRIKAZ"
log_info " Log soubor      : $SIM_LOG"
log_info "================================================================="
echo ""

START_TIME=$(date +%s)

# Spuštění výpočtu přesně podle zadaného formátu
$SPUSTECI_PRIKAZ 2>&1 | tee "$SIM_LOG"
SIM_EXIT_CODE="${PIPESTATUS[0]}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

if [ "$SIM_EXIT_CODE" -ne 0 ]; then
    log_error "Simulace skončila chybou (kód: $SIM_EXIT_CODE)! Doba běhu: ${HOURS}h ${MINS}m ${SECS}s"
    log_error "Zkontrolujte výpis v logu: $SIM_LOG"
    exit "$SIM_EXIT_CODE"
fi

echo ""
log_success "Simulace iPIC3D byla úspěšně dokončena! (Čas: ${HOURS}h ${MINS}m ${SECS}s)"
echo ""

# Návrat do původního adresáře
cd "$START_DIR"

# ------------------------------------------------------------------------------
# KROK 5: Spuštění postprocessingu pro vygenerování grafů
# ------------------------------------------------------------------------------
log_info "================================================================="
log_info " SPOUŠTÍM POSTPROCESSING"
log_info " Zdrojová data: $DATA_DIR"
log_info " Cílová složka: $RESULTS_DIR"
log_info " Kalibrace INP: ${INPUT_DIR}/${INP_FILENAME}"
log_info "================================================================="

# Automatická aktivace virtuálního prostředí Pythonu (pokud existuje)
if [ -d "$VENV_PATH" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    log_info "Aktivuji Python virtuální prostředí z: $VENV_PATH"
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

# Zajištění nalezení balíčku post_process v PYTHONPATH
if [ -d "${START_DIR}/post_process" ]; then
    export PYTHONPATH="${START_DIR}:${PYTHONPATH:-}"
elif [ -d "${SCRIPT_DIR}/post_process" ]; then
    export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
elif [ -d "${HOME}/PIC_3D/post_process" ]; then
    export PYTHONPATH="${HOME}/PIC_3D:${PYTHONPATH:-}"
fi

POST_LOG="${LOG_DIR}/post_${TIMESTAMP}.log"

# Spuštění modulu post_process
$PYTHON_BIN -m post_process \
    -d "$DATA_DIR" \
    -o "$RESULTS_DIR" \
    --inp "${INPUT_DIR}/${INP_FILENAME}" \
    --clean-cache 2>&1 | tee "$POST_LOG"

POST_EXIT_CODE="${PIPESTATUS[0]}"

if [ "$POST_EXIT_CODE" -ne 0 ]; then
    log_error "Postprocessing skončil chybou (kód: $POST_EXIT_CODE)! Zkontrolujte: $POST_LOG"
    exit "$POST_EXIT_CODE"
fi

echo ""
log_success "================================================================="
log_success " Celý proces proběhl úspěšně!"
log_success " Vygenerované grafy jsou uloženy v: $RESULTS_DIR"
log_success "================================================================="
echo ""
