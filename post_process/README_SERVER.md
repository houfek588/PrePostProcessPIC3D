# iPIC3D Post-Processing CLI (Server Edition)

Tento balíček slouží k automatizovanému zpracování a vizualizaci surových dat z částicového řešiče iPIC3D přímo na výpočetním serveru nebo clusteru.

Balíček je **zcela soběstačný** – k jeho běhu stačí pouze tato složka `post_process/`.

---

## 1. Instalace na serveru

### Krok A: Zkopírování na server
Zkopírujte složku `post_process/` na svůj server pomocí `scp` nebo `rsync`:
```bash
scp -r post_process user@cluster.cz:~/ipic3d_post/
```

### Krok B: Virtuální prostředí a závislosti
Na serveru si vytvořte virtuální prostředí a nainstalujte potřebné knihovny:
```bash
cd ~/ipic3d_post/
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Použití z příkazové řádky (CLI)

Balíček lze spustit dvěma rovnocennými způsoby:
```bash
python -m post_process [přepínače]
# nebo
python post_process/cli.py [přepínače]
```

### A. Rychlé spuštění se zadáním parametrů
```bash
python -m post_process \
  -d /path/to/simulation/data/ \
  -o /path/to/plots/ \
  -v Efield_x energy_ele rhoe0
```

### B. Spuštění s načtením JSON konfigurace
Pokud máte připravený `config.json`:
```bash
python -m post_process -c config.json
```

### C. Kombinace: JSON konfigurace přepsaná z příkazové řádky
CLI přepínače mají vyšší prioritu než hodnoty v JSON souboru:
```bash
python -m post_process -c config.json -o /scratch/special_plots/ -v velocity_1_x
```

### D. Automatická kalibrace fyzikálních jednotek ze souboru `.inp`
Pro přesné přepočty os do SI jednotek přímo podle parametrů konkrétní simulace:
```bash
python -m post_process \
  -d /data/sim01/ \
  -o /data/sim01/results/ \
  --inp /data/sim01/ele_beam.inp \
  -v Efield_x
```

### E. Výpis všech podporovaných veličin
```bash
python -m post_process --list-vars
```

---

## 3. Přehled CLI parametrů

| Přepínač | Dlouhý název | Popis | Výchozí hodnota |
|---|---|---|---|
| `-d` | `--data-dir` | Cesta ke složce se surovými daty (VTK, HDF5, ConservedQuantities) | `./` |
| `-o` | `--output-dir` | Cesta ke složce pro uložení výsledných `.png` grafů | `./result/` |
| `-c` | `--config` | Cesta k JSON souboru s nastavením | `config.json` (pokud existuje) |
| `-p` | `--plot-config` | Cesta k JSON souboru pro výběr grafů | `plot_config.json` |
| `-v` | `--vars` | Seznam veličin pro samostatné zpracování | `["Efield_x", "energy_ele"]` |
| `-m` | `--multi-vars` | Seznam veličin pro sdružené zpracování | `[]` |
| | `--plot-types` | Rychlé typy grafů (`val,fft,hist,val2d,fft2d`) | Dle konfigurace |
| | `--inp` | Cesta k `.inp` souboru simulace pro fyzikální škály | Standardní normalizace |
| | `--clean-cache` | Promazat dočasné `.npy` soubory před analýzou | `False` |
| | `--show` | Interaktivní zobrazení oken grafů (lokální běh) | `False` (headless pro server) |
| | `--no-save` | Vypnout ukládání `.png` grafů na disk | Ukládá se |
| | `--list-vars` | Vypsat dostupné veličiny a skončit | |

---

## 4. Příklad skriptu pro SLURM cluster

Uložte jako `run_postprocess.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=ipic_post
#SBATCH --output=post_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

source ~/ipic3d_post/venv/bin/activate

SIM_DIR=/scratch/my_simulation/run01
OUT_DIR=/scratch/my_simulation/run01/plots

python -m post_process \
  --data-dir $SIM_DIR/data/ \
  --output-dir $OUT_DIR \
  --inp $SIM_DIR/simulation.inp \
  --vars Efield_x rhoe0 energy_ele \
  --clean-cache

echo "Hotovo! Grafy jsou v $OUT_DIR"
```
