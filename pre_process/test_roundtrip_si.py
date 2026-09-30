#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_roundtrip_si.py - Ověřovací test obousměrného převodu SI -> .inp -> SI.

Tento test:
1. Vezme zadané fyzikální parametry v SI (sluneční vítr a elektronový svazek).
2. Převede je na bezrozměrné simulační veličiny a vytvoří validní iPIC3D .inp soubor.
3. Načte vygenerovaný .inp soubor pomocí převodníku inp_to_si.py a převede parametry zpět do SI.
4. Porovná výstupní SI hodnoty se vstupními SI hodnotami a ověří jejich shodu.
5. Uloží vygenerovaný textový protokol do .txt souboru.

Spuštění:
    python test_roundtrip_si.py
    nebo
    pytest test_roundtrip_si.py
"""

import os
import sys
import math
import unittest
import gen_inp_file
import inp_to_si

# =========================================================================
# 1. ZADANÉ VSTUPNÍ HODNOTY V SI JEDNOTKÁCH
# =========================================================================
TEST_INPUTS_SI = {
    "B_nT": 10.0,              # Magnetická indukce: B = 10 nT
    "n_i_cm3": 10.0,           # Hustota iontů: n_i = 10 cm^-3
    "T_i_eV": 10.0,            # Teplota iontů: T_i = 10 eV
    "n_e_total_cm3": 10.0,     # Celková hustota elektronů: n_e = 10 cm^-3
    "T_e_eV": 10.0,            # Teplota elektronů: T_e = 10 eV
    "n_b_cm3": 0.01,           # Hustota svazku: n_b = 0.01 cm^-3
    "T_b_eV": 3.0,             # Teplota svazku: T_b = 3.0 eV
    "v_b_factor": 5.0,         # Rychlost svazku: v_b = 5 * v_th
}

# Fyzikální konstanty
CONST_C   = inp_to_si.CONST_C
CONST_E   = inp_to_si.CONST_E
CONST_M_P = inp_to_si.CONST_M_P
CONST_EPS = inp_to_si.CONST_EPS_0
CONST_KB  = inp_to_si.CONST_K_B

# V kinetickém řešiči iPIC3D je standardní poměr hmotností mi / me = 64
MASS_RATIO = 64.0
CONST_M_E  = CONST_M_P / MASS_RATIO


def build_inp_from_si(si_params: dict, inp_filename: str) -> dict:
    """
    Převede zadané SI hodnoty do simulačních jednotek a vygeneruje .inp soubor.
    Vrací slovník s přesnými teoretickými hodnotami pro kontrolu.
    """
    B_si = si_params["B_nT"] * 1.0e-9
    n_i_si = si_params["n_i_cm3"] * 1.0e6
    n_b_si = si_params["n_b_cm3"] * 1.0e6
    n_e_bg_si = (si_params["n_e_total_cm3"] - si_params["n_b_cm3"]) * 1.0e6

    T_i_ev = si_params["T_i_eV"]
    T_e_ev = si_params["T_e_eV"]
    T_b_ev = si_params["T_b_eV"]

    # 1. Výpočet charakteristických rychlostí v SI
    # v_th = sqrt(kT / m) = sqrt(e * T_eV / m)
    vth_i_si = math.sqrt((CONST_E * T_i_ev) / CONST_M_P)
    vth_e_si = math.sqrt((CONST_E * T_e_ev) / CONST_M_E)
    vth_b_si = math.sqrt((CONST_E * T_b_ev) / CONST_M_E)
    vb_si = si_params["v_b_factor"] * vth_b_si

    # 2. Iontové normalizační škály (reference n0 = n_i)
    n0 = n_i_si
    om_pi = math.sqrt((n0 * CONST_E**2) / (CONST_EPS * CONST_M_P))
    d_i = CONST_C / om_pi
    B0 = (CONST_M_P * om_pi) / CONST_E

    # 3. Bezrozměrné simulační parametry
    B_sim = B_si / B0
    rho_i_sim = n_i_si / n0         # 1.0
    rho_b_sim = n_b_si / n0         # 0.001
    rho_e_sim = n_e_bg_si / n0      # 0.999

    vth_i_sim = vth_i_si / CONST_C
    vth_e_sim = vth_e_si / CONST_C
    vth_b_sim = vth_b_si / CONST_C
    vb_sim = vb_si / CONST_C

    # 4. Generování .inp souboru
    # 3 částicové druhy: 0 = background electrons, 1 = protons, 2 = beam electrons
    f = gen_inp_file.PIC3DInputFile(inp_filename, "RoundTripTest", 3)
    f.set_case("none")
    f.set_time_par(0.0001, 80000)
    f.set_mag_field(B_sim, 0.0, 0.0)
    f.set_box_size(0.5, 0.1, 0.1)
    f.set_num_cells(4096, 1, 1)
    f.set_mpi_topology(32, 1, 1)

    # Druh 0: elektrony pozadí
    f.set_part_par(0, -MASS_RATIO, 0.0, rho_e_sim)
    f.set_thermal_vel(0, vth_e_sim, vth_e_sim, vth_e_sim)
    f.set_drift_vel(0, 0.0, 0.0, 0.0)
    f.set_npcell_vel(0, 50, 1, 1)

    # Druh 1: ionty (protony)
    f.set_part_par(1, 1.0, 0.0, rho_i_sim)
    f.set_thermal_vel(1, vth_i_sim, vth_i_sim, vth_i_sim)
    f.set_drift_vel(1, 0.0, 0.0, 0.0)
    f.set_npcell_vel(1, 50, 1, 1)

    # Druh 2: elektronový svazek
    f.set_part_par(2, -MASS_RATIO, 0.0, rho_b_sim)
    f.set_thermal_vel(2, vth_b_sim, vth_b_sim, vth_b_sim)
    f.set_drift_vel(2, vb_sim, 0.0, 0.0)
    f.set_npcell_vel(2, 30, 1, 1)

    with open(inp_filename, "w", encoding="utf-8") as out:
        f.get_file_text(out)

    return {
        "B_nT": si_params["B_nT"],
        "n_i_cm3": si_params["n_i_cm3"],
        "T_i_eV": T_i_ev,
        "n_e_bg_cm3": n_e_bg_si * 1e-6,
        "n_e_total_cm3": si_params["n_e_total_cm3"],
        "T_e_eV": T_e_ev,
        "n_b_cm3": si_params["n_b_cm3"],
        "T_b_eV": T_b_ev,
        "vth_b_si": vth_b_si,
        "v_b_si": vb_si,
        "v_b_factor": si_params["v_b_factor"],
        "B_sim": B_sim,
        "rho_i_sim": rho_i_sim,
        "rho_e_sim": rho_e_sim,
        "rho_b_sim": rho_b_sim,
        "vth_i_sim": vth_i_sim,
        "vth_e_sim": vth_e_sim,
        "vth_b_sim": vth_b_sim,
        "vb_sim": vb_sim,
    }


def run_roundtrip_test():
    """Spustí kompletní ověřovací test obousměrného převodu."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_inp = os.path.join(script_dir, "test_roundtrip.inp")
    test_out_txt = os.path.join(script_dir, "test_roundtrip_si_parameters.txt")

    sep = "=" * 90
    subsep = "-" * 90

    print(sep)
    print(" TEST OBOUSTRANNÉHO PŘEVODU: SI -> .inp -> SI")
    print(sep)

    # 1. KROK: Vytvoření .inp souboru ze zadaných SI hodnot
    print(f"\n[KROK 1] Generuji .inp soubor ze zadaných SI hodnot...")
    expected = build_inp_from_si(TEST_INPUTS_SI, test_inp)
    print(f"  Vytvořen soubor: {test_inp}")
    print(f"  Simulační parametry zapsané do .inp:")
    print(f"    B_sim = {expected['B_sim']:.8g}")
    print(f"    rho_e = {expected['rho_e_sim']:.6g},  rho_i = {expected['rho_i_sim']:.6g},  rho_b = {expected['rho_b_sim']:.6g}")
    print(f"    uth_e = {expected['vth_e_sim']:.8g},  uth_i = {expected['vth_i_sim']:.8g},  uth_b = {expected['vth_b_sim']:.8g}")
    print(f"    u0_b  = {expected['vb_sim']:.8g}")

    # 2. KROK: Zpětný převod z .inp do SI pomocí inp_to_si.py
    print(f"\n[KROK 2] Načítám vygenerovaný .inp soubor a převádím zpět do SI...")
    converter = inp_to_si.InpToSIConverter(test_inp, ref_density_m3=TEST_INPUTS_SI["n_i_cm3"] * 1e6)
    report_text = converter.generate_report()
    si_result = converter.get_si_data()

    # Uložení textového protokolu
    with open(test_out_txt, "w", encoding="utf-8") as f_out:
        f_out.write(report_text)
    print(f"  Protokol uložen do: {test_out_txt}")

    # Extrakce zpětně napočítaných hodnot
    sp_e = si_result["species"][0]
    sp_i = si_result["species"][1]
    sp_b = si_result["species"][2]

    actual = {
        "B_nT": si_result["B_tot_nT"],
        "n_i_cm3": sp_i["conc_cm3"],
        "T_i_eV": sp_i["T_x_ev"],
        "n_e_bg_cm3": sp_e["conc_cm3"],
        "n_e_total_cm3": sp_e["conc_cm3"] + sp_b["conc_cm3"],
        "T_e_eV": sp_e["T_x_ev"],
        "n_b_cm3": sp_b["conc_cm3"],
        "T_b_eV": sp_b["T_x_ev"],
        "v_b_si": sp_b["u0_x_si"],
        "vth_b_si": sp_b["vth_x_si"],
        "v_b_factor": sp_b["u0_x_si"] / sp_b["vth_x_si"] if sp_b["vth_x_si"] > 0 else 0.0,
    }

    # 3. KROK: Porovnání hodnot a vyhodnocení
    print(f"\n[KROK 3] Kontrola shody vstupních a zpětně vypočtených SI parametrů:")
    print(subsep)
    print(f"{'Fyzikální veličina':<30} | {'Zadáno v SI':<16} | {'Zpětně ze sim':<16} | {'Rel. chyba':<12} | {'Stav'}")
    print(subsep)

    checks = [
        ("Magnetická indukce B [nT]", TEST_INPUTS_SI["B_nT"], actual["B_nT"]),
        ("Hustota iontů n_i [cm^-3]", TEST_INPUTS_SI["n_i_cm3"], actual["n_i_cm3"]),
        ("Teplota iontů T_i [eV]", TEST_INPUTS_SI["T_i_eV"], actual["T_i_eV"]),
        ("Hustota elektronů pozadí [cm^-3]", expected["n_e_bg_cm3"], actual["n_e_bg_cm3"]),
        ("Celková hustota elektronů [cm^-3]", TEST_INPUTS_SI["n_e_total_cm3"], actual["n_e_total_cm3"]),
        ("Teplota elektronů T_e [eV]", TEST_INPUTS_SI["T_e_eV"], actual["T_e_eV"]),
        ("Hustota svazku n_b [cm^-3]", TEST_INPUTS_SI["n_b_cm3"], actual["n_b_cm3"]),
        ("Teplota svazku T_b [eV]", TEST_INPUTS_SI["T_b_eV"], actual["T_b_eV"]),
        ("Rychlost svazku v_b [m/s]", expected["v_b_si"], actual["v_b_si"]),
        ("Tepelná rychlost svazku [m/s]", expected["vth_b_si"], actual["vth_b_si"]),
        ("Poměr v_b / v_th (násobek)", TEST_INPUTS_SI["v_b_factor"], actual["v_b_factor"]),
    ]

    all_passed = True
    tolerance = 1.0e-5  # Požadovaná relativní přesnost (0.001 %)

    for name, expected_val, actual_val in checks:
        if expected_val != 0:
            rel_err = abs(actual_val - expected_val) / abs(expected_val)
        else:
            rel_err = abs(actual_val - expected_val)

        is_ok = rel_err < tolerance
        if not is_ok:
            all_passed = False

        status = "PASSED (OK)" if is_ok else "FAILED (CHYBA)"
        print(f"{name:<30} | {expected_val:<16.6g} | {actual_val:<16.6g} | {rel_err:<12.2e} | {status}")

    print(subsep)

    if all_passed:
        print("\n>>> VŠECHNY TESTY ÚSPĚŠNĚ PROŠLY! (100% shoda všech fyzikálních veličin) <<<")
    else:
        print("\n>>> CHYBA: Některé parametry se neshodují! <<<")

    print(sep)
    return all_passed


class TestSIConversionRoundTrip(unittest.TestCase):
    """Integrace se standardním modulem unittest (bez nutnosti instalovat pytest)."""
    def test_roundtrip_conversion(self):
        self.assertTrue(run_roundtrip_test())


def test_roundtrip():
    """Pytest kompatibilní testovací funkce."""
    assert run_roundtrip_test() is True


if __name__ == "__main__":
    success = run_roundtrip_test()
    sys.exit(0 if success else 1)
