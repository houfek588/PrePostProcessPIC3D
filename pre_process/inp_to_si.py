#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inp_to_si.py - Převodník vstupních parametrů iPIC3D (.inp) do jednotek SI.

Tento skript načte simulační vstupní soubor (.inp), převede všechny bezrozměrné
fyzikální parametry do mezinárodní soustavy jednotek (SI) a uloží podrobný
a strukturovaný výpis do textového souboru (.txt) i do konzole.

Použití:
    python inp_to_si.py [soubor.inp] [-o vystup.txt] [--density N0] [--bfield B_SI]

Příklady:
    python inp_to_si.py myfile.inp
    python inp_to_si.py myfile.inp -o parametry_si.txt
    python inp_to_si.py ../../res_data_vth/beam_hot_tenuous_few/ele_beam_bump_on_tail_optimized.inp
    python inp_to_si.py myfile.inp --density 10.0   # n0 = 10 cm^-3
"""

import sys
import os
import math
import re
import argparse
from datetime import datetime

# =========================================================================
# ZÁKLADNÍ FYZIKÁLNÍ KONSTANTY (CODATA 2018 / SI)
# =========================================================================
CONST_C      = 2.99792458e8         # Rychlost světla ve vakuu [m/s]
CONST_E      = 1.602176634e-19      # Elementární náboj [C]
CONST_M_P    = 1.67262192595e-27    # Hmotnost protonu [kg]
CONST_M_E_SI = 9.1093837139e-31     # Skutečná fyzikální hmotnost elektronu [kg]
CONST_EPS_0  = 8.8541878128e-12     # Vakuová permitivita [F/m = C^2 / (N m^2)]
CONST_MU_0   = 1.25663706212e-6     # Vakuová permeabilita [N / A^2 = H/m]
CONST_K_B    = 1.380649e-23         # Boltzmannova konstanta [J/K]

# Výchozí referenční hustota plazmatu slunečního větru na 1 AU:
DEFAULT_N0_M3 = 1.0e7               # 10 cm^-3 = 1.0e7 m^-3


def parse_inp_file(filepath: str) -> dict:
    """
    Načte a naparsuje iPIC3D .inp soubor do slovníku klíč-hodnota.
    Odstraní komentáře za znakem '#' a převede číselné hodnoty.
    """
    params = {}
    raw_lines = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            raw_lines.append(line)
            clean = line.strip()
            if not clean or clean.startswith('#'):
                continue
            if '#' in clean:
                clean = clean[:clean.index('#')].strip()
            if '=' in clean:
                key, val = clean.split('=', 1)
                key = key.strip()
                val = val.strip()
                tokens = val.split()
                parsed = []
                for tok in tokens:
                    try:
                        if '.' in tok or 'e' in tok.lower():
                            parsed.append(float(tok))
                        else:
                            parsed.append(int(tok))
                    except ValueError:
                        parsed.append(tok)
                if len(parsed) == 1:
                    params[key] = parsed[0]
                elif len(parsed) > 1:
                    params[key] = parsed
                else:
                    params[key] = ''

    return params, raw_lines


def try_detect_reference_from_comments(raw_lines: list):
    """
    Zkusí z komentářů v hlavičce detekovat zadanou hustotu n_i [cm-3] nebo B [nT].
    """
    detected_n0 = None
    detected_B = None
    
    for line in raw_lines:
        # hledá např. n_i = 10 cm-3 nebo n_e = 10 cm-3
        m_n = re.search(r'n_[ie]\s*=\s*([0-9\.]+)\s*cm\-?3', line, re.IGNORECASE)
        if m_n and detected_n0 is None:
            detected_n0 = float(m_n.group(1)) * 1.0e6  # cm^-3 -> m^-3
        
        # hledá např. B = 10 nT
        m_b = re.search(r'B\s*=\s*([0-9\.]+)\s*nT', line, re.IGNORECASE)
        if m_b and detected_B is None:
            detected_B = float(m_b.group(1)) * 1.0e-9  # nT -> T

    return detected_n0, detected_B


def to_list(val, expected_len=None, default=0.0):
    """Zajistí, že hodnota je seznam dané délky."""
    if val is None:
        lst = []
    elif isinstance(val, (list, tuple)):
        lst = list(val)
    else:
        lst = [val]
    if expected_len is not None:
        if len(lst) < expected_len:
            lst = lst + [default] * (expected_len - len(lst))
        elif len(lst) > expected_len:
            lst = lst[:expected_len]
    return lst


class InpToSIConverter:
    """Třída provádějící fyzikální převod vstupů iPIC3D do soustavy SI."""

    def __init__(self, inp_filepath: str, ref_density_m3: float = None, ref_bfield_tesla: float = None):
        self.inp_filepath = os.path.abspath(inp_filepath)
        self.params, self.raw_lines = parse_inp_file(self.inp_filepath)
        
        # Určení referenční škály
        det_n0, det_b = try_detect_reference_from_comments(self.raw_lines)
        
        if ref_density_m3 is not None:
            self.n0 = float(ref_density_m3)
            self.ref_source = f"uživatelsky zadaná hustota pozadí ({self.n0*1e-6:.2f} cm^-3)"
        elif ref_bfield_tesla is not None:
            B_sim_x = float(self.params.get('B0x', 0.0))
            B_sim_y = float(self.params.get('B0y', 0.0))
            B_sim_z = float(self.params.get('B0z', 0.0))
            B_sim_tot = math.sqrt(B_sim_x**2 + B_sim_y**2 + B_sim_z**2)
            if B_sim_tot > 0:
                B0_scale = ref_bfield_tesla / B_sim_tot
                om_pi = (CONST_E * B0_scale) / CONST_M_P
                self.n0 = (CONST_EPS_0 * CONST_M_P * om_pi**2) / (CONST_E**2)
                self.ref_source = f"odvozeno z B = {ref_bfield_tesla*1e9:.2f} nT (n0 = {self.n0*1e-6:.2f} cm^-3)"
            else:
                self.n0 = DEFAULT_N0_M3
                self.ref_source = f"výchozí hodnota pro sluneční vítr ({self.n0*1e-6:.2f} cm^-3)"
        elif det_n0 is not None:
            self.n0 = det_n0
            self.ref_source = f"detekováno z komentáře souboru ({self.n0*1e-6:.2f} cm^-3)"
        elif det_b is not None:
            B_sim_x = float(self.params.get('B0x', 0.0))
            B_sim_y = float(self.params.get('B0y', 0.0))
            B_sim_z = float(self.params.get('B0z', 0.0))
            B_sim_tot = math.sqrt(B_sim_x**2 + B_sim_y**2 + B_sim_z**2)
            if B_sim_tot > 0:
                B0_scale = det_b / B_sim_tot
                om_pi = (CONST_E * B0_scale) / CONST_M_P
                self.n0 = (CONST_EPS_0 * CONST_M_P * om_pi**2) / (CONST_E**2)
                self.ref_source = f"odvozeno z komentáře B = {det_b*1e9:.2f} nT (n0 = {self.n0*1e-6:.2f} cm^-3)"
            else:
                self.n0 = DEFAULT_N0_M3
                self.ref_source = f"výchozí hodnota pro sluneční vítr ({self.n0*1e-6:.2f} cm^-3)"
        else:
            self.n0 = DEFAULT_N0_M3
            self.ref_source = f"výchozí referenční hodnota pro sluneční vítr ({self.n0*1e-6:.2f} cm^-3)"

        # Výpočet základních normalizačních škál
        self._calculate_normalization_scales()

    def _calculate_normalization_scales(self):
        """Vypočítá charakteristické iontové škály pro normalizaci."""
        self.om_pi = math.sqrt((self.n0 * CONST_E**2) / (CONST_EPS_0 * CONST_M_P))
        self.f_pi = self.om_pi / (2.0 * math.pi)
        self.t0 = 1.0 / self.om_pi                        # [s] Časová škála
        self.d_i = CONST_C / self.om_pi                   # [m] Ion skin depth
        self.v0 = CONST_C                                 # [m/s] Rychlostní škála
        self.m0 = CONST_M_P                               # [kg] Hmotnostní škála
        self.q0 = CONST_E                                 # [C] Nábojová škála
        self.B0 = (CONST_M_P * self.om_pi) / CONST_E      # [T] Škála magnetického pole
        self.E0 = CONST_C * self.B0                       # [V/m] Škála elektrického pole
        self.energy0 = CONST_M_P * CONST_C**2             # [J] Škála energie

    def generate_report(self) -> str:
        """Sestaví kompletní podrobný protokol o převodu do SI."""
        p = self.params
        lines = []
        sep = "=" * 80
        subsep = "-" * 80

        lines.append(sep)
        lines.append(" PŘEVOD VSTUPNÍCH PARAMETRŮ iPIC3D DO SOUSTAVY JEDNOTEK SI")
        lines.append(sep)
        lines.append(f" Vstupní soubor : {self.inp_filepath}")
        lines.append(f" Datum převodu  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f" Základní škála : n_0 = {self.n0:.4e} m^-3 ({self.n0*1e-6:.2f} cm^-3)")
        lines.append(f" Zdroj reference: {self.ref_source}")
        lines.append(sep)

        # 1. NORMALIZAČNÍ ŠKÁLY
        lines.append("\n[1] ZÁKLADNÍ NORMALIZAČNÍ ŠKÁLY (IONTOVÉ PARAMETRY)")
        lines.append(subsep)
        lines.append(f"  Rychlost světla [c]          : {CONST_C:.6e} m/s  (v_sim = 1.0)")
        lines.append(f"  Iontová plazmová frekvence   : omega_pi = {self.om_pi:.4f} rad/s (f_pi = {self.f_pi:.2f} Hz)")
        lines.append(f"  Časová perioda iontů [1/w_pi]: t_0 = {self.t0:.6e} s ({self.t0*1e3:.4f} ms)")
        lines.append(f"  Ion skin depth (škála délky) : d_i = {self.d_i:.4f} m ({self.d_i/1000.0:.3f} km)")
        lines.append(f"  Škála magnetického pole [B_0]: B_0 = {self.B0:.6e} T ({self.B0*1e9:.2f} nT)")
        lines.append(f"  Škála elektrického pole [E_0]: E_0 = {self.E0:.4f} V/m")
        lines.append(f"  Škála hustoty plazmatu [n_0] : n_0 = {self.n0:.4e} m^-3 ({self.n0*1e-6:.2f} cm^-3)")
        lines.append(f"  Referenční iontová hmotnost  : m_i = {CONST_M_P:.6e} kg (proton)")
        lines.append(f"  Elementární náboj            : e   = {CONST_E:.6e} C")

        # 2. ČASOVÝ KROK A DÉLKA SIMULACE
        dt_sim = float(p.get('dt', 0.0))
        ncycles = int(p.get('ncycles', 0))
        dt_si = dt_sim * self.t0
        total_time_si = ncycles * dt_si

        field_cycle = int(p.get('FieldOutputCycle', 0))
        part_cycle = int(p.get('ParticlesOutputCycle', 0))

        lines.append("\n[2] ČASOVÉ PARAMETRY SIMULACE")
        lines.append(subsep)
        lines.append(f"  Časový krok (simulace)       : dt = {dt_sim:.6g} [1/omega_pi]")
        lines.append(f"  Časový krok (SI)             : dt_SI = {dt_si:.6e} s ({dt_si*1e9:.3f} ns)")
        lines.append(f"  Počet časových cyklů         : ncycles = {ncycles}")
        lines.append(f"  Celková doba simulace (SI)   : T_total = {total_time_si:.6e} s ({total_time_si*1e3:.4f} ms)")
        lines.append(f"  Celkový počet iont. period   : {ncycles * dt_sim:.3f} [1/omega_pi]")
        
        if field_cycle > 0:
            dt_field_si = field_cycle * dt_si
            num_field_snaps = ncycles // field_cycle
            lines.append(f"  Výstup polí (FieldOutputCycle): každých {field_cycle} cyklů = {dt_field_si*1e6:.3f} us ({num_field_snaps} snímků)")
        if part_cycle > 0:
            dt_part_si = part_cycle * dt_si
            num_part_snaps = ncycles // part_cycle
            lines.append(f"  Výstup částic (PartOutputCycle): každých {part_cycle} cyklů = {dt_part_si*1e3:.4f} ms ({num_part_snaps} snímků)")

        # 3. PROSTOROVÁ OBLAST A SÍŤ
        Lx_sim = float(p.get('Lx', 1.0))
        Ly_sim = float(p.get('Ly', 1.0))
        Lz_sim = float(p.get('Lz', 1.0))

        nxc = int(p.get('nxc', 1))
        nyc = int(p.get('nyc', 1))
        nzc = int(p.get('nzc', 1))

        Lx_si = Lx_sim * self.d_i
        Ly_si = Ly_sim * self.d_i
        Lz_si = Lz_sim * self.d_i

        dx_si = Lx_si / nxc
        dy_si = Ly_si / nyc
        dz_si = Lz_si / nzc

        lines.append("\n[3] GEOMETRIE BOXU A DISKRETIZACE SÍTĚ")
        lines.append(subsep)
        lines.append(f"  Rozměry boxu (simulace)      : Lx = {Lx_sim:.4g} d_i,  Ly = {Ly_sim:.4g} d_i,  Lz = {Lz_sim:.4g} d_i")
        lines.append(f"  Rozměry boxu (SI - metry)    : Lx = {Lx_si:.3f} m,  Ly = {Ly_si:.3f} m,  Lz = {Lz_si:.3f} m")
        lines.append(f"  Rozměry boxu (SI - kilometry): Lx = {Lx_si/1000.0:.4f} km,  Ly = {Ly_si/1000.0:.4f} km,  Lz = {Lz_si/1000.0:.4f} km")
        lines.append(f"  Objem simulačního boxu       : V_box = {Lx_si*Ly_si*Lz_si:.4e} m^3")
        lines.append(f"  Počet buněk sítě (nxc,nyc,nzc): {nxc} x {nyc} x {nzc}  (celkem {nxc*nyc*nzc} buněk)")
        lines.append(f"  Velikost buňky dx, dy, dz    : dx = {dx_si:.4f} m,  dy = {dy_si:.4f} m,  dz = {dz_si:.4f} m")
        
        # MPI a okraje
        xlen = p.get('XLEN', 1)
        ylen = p.get('YLEN', 1)
        zlen = p.get('ZLEN', 1)
        lines.append(f"  MPI domény (XLEN, YLEN, ZLEN): {xlen} x {ylen} x {zlen}")
        lines.append(f"  Periodické okraje (X, Y, Z)  : PERIODICX={p.get('PERIODICX', 1)}, PERIODICY={p.get('PERIODICY', 1)}, PERIODICZ={p.get('PERIODICZ', 1)}")

        # 4. MAGNETICKÉ POLE
        B0x_sim = float(p.get('B0x', 0.0))
        B0y_sim = float(p.get('B0y', 0.0))
        B0z_sim = float(p.get('B0z', 0.0))
        B_tot_sim = math.sqrt(B0x_sim**2 + B0y_sim**2 + B0z_sim**2)

        B0x_si = B0x_sim * self.B0
        B0y_si = B0y_sim * self.B0
        B0z_si = B0z_sim * self.B0
        B_tot_si = B_tot_sim * self.B0

        lines.append("\n[4] MAGNETICKÉ POLE")
        lines.append(subsep)
        lines.append(f"  B (simulace) [Bx, By, Bz]    : [{B0x_sim:.6g}, {B0y_sim:.6g}, {B0z_sim:.6g}]  (|B| = {B_tot_sim:.6g})")
        lines.append(f"  B (SI - Tesla)               : [{B0x_si:.6e}, {B0y_si:.6e}, {B0z_si:.6e}] T")
        lines.append(f"  B (SI - nanotesla)           : [{B0x_si*1e9:.3f}, {B0y_si*1e9:.3f}, {B0z_si*1e9:.3f}] nT  (|B| = {B_tot_si*1e9:.3f} nT)")
        lines.append(f"  Alfvénova rychlost (v_A/c)   : v_A / c = |B_sim| = {B_tot_sim:.6g}  (v_A = {B_tot_sim * CONST_C:.3f} m/s = {B_tot_sim * CONST_C / 1000.0:.3f} km/s)")

        # 5. ČÁSTICOVÉ DRUHY
        ns = int(p.get('ns', 1))
        qom_list = to_list(p.get('qom', 1.0), ns)
        rho_init_list = to_list(p.get('rhoINIT', 1.0), ns)
        uth_list = to_list(p.get('uth', 0.0), ns)
        vth_list = to_list(p.get('vth', 0.0), ns)
        wth_list = to_list(p.get('wth', 0.0), ns)
        u0_list = to_list(p.get('u0', 0.0), ns)
        v0_list = to_list(p.get('v0', 0.0), ns)
        w0_list = to_list(p.get('w0', 0.0), ns)
        npcelx_list = to_list(p.get('npcelx', 1), ns)
        npcely_list = to_list(p.get('npcely', 1), ns)
        npcelz_list = to_list(p.get('npcelz', 1), ns)

        lines.append("\n[5] PARAMETRY JEDNOTLIVÝCH DRUHŮ ČÁSTIC")
        lines.append(subsep)

        total_charge_density_si = 0.0
        total_thermal_pressure = 0.0
        electron_debye_length = None
        species_records = []

        for s in range(ns):
            qom = float(qom_list[s])
            rho_init = float(rho_init_list[s])
            
            # Odvození náboje a hmotnosti z qom = (q/e) / (m/m_i)
            # Standardně pro protony qom = 1.0, pro elektrony qom = -mi/me
            if qom > 0:
                q_s = CONST_E
                m_s = CONST_M_P / qom
                mass_ratio = qom
                species_type = "Ionty (protony)"
            else:
                q_s = -CONST_E
                m_s = CONST_M_P / abs(qom)
                mass_ratio = abs(qom)
                species_type = "Elektrony"
                if abs(float(u0_list[s])) > 0 or abs(float(v0_list[s])) > 0 or abs(float(w0_list[s])) > 0:
                    species_type = "Elektronový svazek"

            # Hustota
            conc_m3 = rho_init * self.n0
            conc_cm3 = conc_m3 * 1.0e-6
            charge_density_s = conc_m3 * q_s
            total_charge_density_si += charge_density_s

            # Rychlosti
            uth_x_sim = float(uth_list[s])
            uth_y_sim = float(vth_list[s])
            uth_z_sim = float(wth_list[s])
            vth_x_si = uth_x_sim * CONST_C
            vth_y_si = uth_y_sim * CONST_C
            vth_z_si = uth_z_sim * CONST_C
            vth_tot_si = math.sqrt(vth_x_si**2 + vth_y_si**2 + vth_z_si**2)

            u0_x_sim = float(u0_list[s])
            u0_y_sim = float(v0_list[s])
            u0_z_sim = float(w0_list[s])
            u0_x_si = u0_x_sim * CONST_C
            u0_y_si = u0_y_sim * CONST_C
            u0_z_si = u0_z_sim * CONST_C
            u0_tot_si = math.sqrt(u0_x_si**2 + u0_y_si**2 + u0_z_si**2)

            # Teploty (kT = m * vth^2)
            # Pozor: v iPIC3D je vth definováno jako směrodatná odchylka jedné složky: vth = sqrt(kT/m)
            T_x_kelvin = (m_s * vth_x_si**2) / CONST_K_B if vth_x_si > 0 else 0.0
            T_x_ev = (m_s * vth_x_si**2) / CONST_E if vth_x_si > 0 else 0.0

            T_iso_kelvin = (m_s * (vth_x_si**2 + vth_y_si**2 + vth_z_si**2) / 3.0) / CONST_K_B if vth_tot_si > 0 else 0.0
            T_iso_ev = (m_s * (vth_x_si**2 + vth_y_si**2 + vth_z_si**2) / 3.0) / CONST_E if vth_tot_si > 0 else 0.0

            total_thermal_pressure += conc_m3 * CONST_K_B * T_iso_kelvin

            # Kinetická energie driftu
            E_drift_joule = 0.5 * m_s * u0_tot_si**2
            E_drift_ev = E_drift_joule / CONST_E

            # Plazmová frekvence
            if conc_m3 > 0 and m_s > 0:
                om_ps = math.sqrt((conc_m3 * q_s**2) / (CONST_EPS_0 * m_s))
                f_ps = om_ps / (2.0 * math.pi)
                period_ps = 1.0 / f_ps if f_ps > 0 else 0.0
                skin_depth_s = CONST_C / om_ps
                debye_len_s = (vth_x_si / om_ps) if vth_x_si > 0 else 0.0
            else:
                om_ps = f_ps = period_ps = skin_depth_s = debye_len_s = 0.0

            if qom < 0 and electron_debye_length is None and debye_len_s > 0:
                electron_debye_length = debye_len_s

            # Cyklotronní frekvence
            if B_tot_si > 0 and m_s > 0:
                om_cs = (abs(q_s) * B_tot_si) / m_s
                f_cs = om_cs / (2.0 * math.pi)
                r_larmor = (vth_x_si / om_cs) if om_cs > 0 else 0.0
            else:
                om_cs = f_cs = r_larmor = 0.0

            # Počet makročástic
            npc_x = int(npcelx_list[s])
            npc_y = int(npcely_list[s])
            npc_z = int(npcelz_list[s])
            npcel_tot = npc_x * npc_y * npc_z
            total_macroparts = npcel_tot * (nxc * nyc * nzc)
            
            # Fyzický počet částic reprezentovaný jednou makročásticí (váha)
            cell_volume_si = dx_si * dy_si * dz_si
            particles_per_cell_physical = conc_m3 * cell_volume_si
            part_weight = particles_per_cell_physical / npcel_tot if npcel_tot > 0 else 0.0

            lines.append(f"\n  Druh #{s}: {species_type} (qom = {qom})")
            lines.append(f"  " + "-" * 50)
            lines.append(f"    Hmotnost částice m_s       : {m_s:.6e} kg  (m_i / m_s = {mass_ratio:.2f})")
            lines.append(f"    Náboj částice q_s          : {q_s:+.6e} C  ({q_s/CONST_E:+.1f} e)")
            lines.append(f"    Normalizovaná hustota rho  : {rho_init:.6g}")
            lines.append(f"    Koncentrace n_s            : {conc_m3:.4e} m^-3  ({conc_cm3:.4f} cm^-3)")
            lines.append(f"    Tepelná rychlost vth (sim) : [{uth_x_sim:.6g}, {uth_y_sim:.6g}, {uth_z_sim:.6g}] c")
            lines.append(f"    Tepelná rychlost vth (SI)  : [{vth_x_si:.2f}, {vth_y_si:.2f}, {vth_z_si:.2f}] m/s  (|vth| = {vth_tot_si:.2f} m/s)")
            lines.append(f"    Tepelná rychlost vth (km/s): [{vth_x_si/1000.0:.3f}, {vth_y_si/1000.0:.3f}, {vth_z_si/1000.0:.3f}] km/s")
            lines.append(f"    Teplota v ose X (T_x)      : {T_x_ev:.3f} eV  ({T_x_kelvin:.1f} K)")
            lines.append(f"    Izotropní teplota (T_iso)  : {T_iso_ev:.3f} eV  ({T_iso_kelvin:.1f} K)")
            lines.append(f"    Driftová rychlost u0 (sim) : [{u0_x_sim:.6g}, {u0_y_sim:.6g}, {u0_z_sim:.6g}] c")
            lines.append(f"    Driftová rychlost u0 (SI)  : [{u0_x_si:.2f}, {u0_y_si:.2f}, {u0_z_si:.2f}] m/s  ({u0_tot_si/1000.0:.3f} km/s)")
            if u0_tot_si > 0:
                lines.append(f"    Kinetická energie driftu   : {E_drift_ev:.4f} eV  ({E_drift_joule:.4e} J)")
                if vth_x_si > 0:
                    lines.append(f"    Poměr drift / termál (u/vth): {u0_x_si / vth_x_si:.3f}")
            lines.append(f"    Plazmová frekvence w_ps    : {om_ps:.4f} rad/s  (f_ps = {f_ps:.2f} Hz, perioda = {period_ps*1e6:.3f} us)")
            if om_cs > 0:
                lines.append(f"    Cyklotronní frekvence w_cs : {om_cs:.4f} rad/s  (f_cs = {f_cs:.2f} Hz)")
                lines.append(f"    Larmorův poloměr r_L       : {r_larmor:.4f} m  ({r_larmor/1000.0:.4f} km)")
            lines.append(f"    Skin depth druhu d_s       : {skin_depth_s:.4f} m  ({skin_depth_s/1000.0:.3f} km)")
            lines.append(f"    Debyeova délka lambda_D    : {debye_len_s:.4f} m")
            lines.append(f"    Makročástice na buňku      : {npc_x} x {npc_y} x {npc_z} = {npcel_tot}")
            lines.append(f"    Celkem makročástic v boxu  : {total_macroparts:,}")
            lines.append(f"    Fyzikální částice / makroč.: {part_weight:.4e} (váha částice)")

            species_records.append({
                'index': s,
                'type': species_type,
                'qom': qom,
                'm_s': m_s,
                'q_s': q_s,
                'rho_init': rho_init,
                'conc_m3': conc_m3,
                'conc_cm3': conc_cm3,
                'vth_x_si': vth_x_si,
                'vth_y_si': vth_y_si,
                'vth_z_si': vth_z_si,
                'vth_tot_si': vth_tot_si,
                'T_x_ev': T_x_ev,
                'T_x_kelvin': T_x_kelvin,
                'T_iso_ev': T_iso_ev,
                'T_iso_kelvin': T_iso_kelvin,
                'u0_x_si': u0_x_si,
                'u0_y_si': u0_y_si,
                'u0_z_si': u0_z_si,
                'u0_tot_si': u0_tot_si,
                'E_drift_ev': E_drift_ev,
                'om_ps': om_ps,
                'f_ps': f_ps,
                'skin_depth_s': skin_depth_s,
                'debye_len_s': debye_len_s,
                'npcel_tot': npcel_tot
            })

        # 6. FYZIKÁLNÍ CHARAKTERISTIKY PLAZMATU A KONTROLA STABILITY
        B_mag_pressure = (B_tot_si**2) / (2.0 * CONST_MU_0)
        plasma_beta = (total_thermal_pressure / B_mag_pressure) if B_mag_pressure > 0 else float('inf')

        lines.append("\n[6] INTEGRÁLNÍ CHARAKTERISTIKY PLAZMATU A STABILITNÍ KRITÉRIA")
        lines.append(subsep)
        lines.append(f"  Celková nábojová hustota     : rho_q = {total_charge_density_si:+.6e} C/m^3")
        charge_neutrality_ratio = abs(total_charge_density_si) / (CONST_E * self.n0)
        if charge_neutrality_ratio < 1e-4:
            lines.append(f"  Kontrola kvazineutrality     : OK (odchylka {charge_neutrality_ratio*100.0:.4f} % z e*n0)")
        else:
            lines.append(f"  Kontrola kvazineutrality     : POZOR! Výrazná nenulová nábojová hustota ({charge_neutrality_ratio*100.0:.2f} %)")

        lines.append(f"  Tepelný tlak plazmatu        : p_thermal = {total_thermal_pressure:.6e} Pa")
        lines.append(f"  Magnetický tlak              : p_mag = {B_mag_pressure:.6e} Pa")
        lines.append(f"  Plazmové beta (p_th / p_mag) : beta = {plasma_beta:.4f}")

        # CFL podmínka
        c_dt_si = CONST_C * dt_si
        lines.append(f"\n  Podmínka rychlosti světla (CFL): c * dt <= dx")
        lines.append(f"    c * dt_SI                  : {c_dt_si:.4f} m  (c * dt_sim = {dt_sim * float(p.get('c', 1.0)):.6g} d_i)")
        lines.append(f"    dx_SI                      : {dx_si:.4f} m  (dx_sim = {Lx_sim / nxc:.6g} d_i)")
        cfl_ratio = c_dt_si / dx_si
        if cfl_ratio <= 1.0:
            lines.append(f"    Výsledek CFL               : SPLNĚNA (c * dt / dx = {cfl_ratio:.3f} <= 1.0)")
        else:
            lines.append(f"    Výsledek CFL               : PŘEKROČENA (c * dt / dx = {cfl_ratio:.3f} > 1.0)")
            lines.append(f"    Poznámka                   : V iPIC3D je překročení explicitní CFL podmínky korektní díky implicitnímu schématu.")

        # Rozlišení Debyeovy délky
        if electron_debye_length and electron_debye_length > 0:
            dx_over_lambda_d = dx_si / electron_debye_length
            lines.append(f"\n  Rozlišení Debyeovy délky elektronů:")
            lines.append(f"    Debyeova délka elektronů   : lambda_De = {electron_debye_length:.4f} m")
            lines.append(f"    Velikost buňky dx_SI       : dx = {dx_si:.4f} m")
            lines.append(f"    Poměr dx / lambda_De       : {dx_over_lambda_d:.3f} lambda_De")
            if dx_over_lambda_d <= 2.0:
                lines.append(f"    Kvalita rozlišení          : VÝBORNÁ (dx <= 2 * lambda_De, plně zachycuje elektrostatické vlny)")
            elif dx_over_lambda_d <= 5.0:
                lines.append(f"    Kvalita rozlišení          : DOBRÁ (vhodné pro implicitní PIC)")
            else:
                lines.append(f"    Kvalita rozlišení          : HRUBÁ (buňka je podstatně větší než Debyeova délka)")

        lines.append("\n" + sep)
        lines.append(" KONEC PROTOKOLU")
        lines.append(sep)

        self.si_data = {
            'B_tot_nT': B_tot_si * 1e9,
            'B_tot_T': B_tot_si,
            'Bx_nT': B0x_si * 1e9,
            'By_nT': B0y_si * 1e9,
            'Bz_nT': B0z_si * 1e9,
            'dt_s': dt_si,
            'T_total_s': total_time_si,
            'Lx_m': Lx_si,
            'Ly_m': Ly_si,
            'Lz_m': Lz_si,
            'dx_m': dx_si,
            'n0_m3': self.n0,
            'n0_cm3': self.n0 * 1e-6,
            'species': species_records,
            'plasma_beta': plasma_beta,
            'cfl_ratio': cfl_ratio
        }

        return "\n".join(lines)

    def get_si_data(self) -> dict:
        """Vrátí strukturovaný slovník se všemi přepočtenými fyzikálními veličinami."""
        if not hasattr(self, 'si_data') or self.si_data is None:
            self.generate_report()
        return self.si_data


def convert_inp_to_si(inp_file: str, out_txt_file: str = None, density_cm3: float = None, bfield_nt: float = None):
    """
    Hlavní obslužná funkce pro načtení .inp souboru, převod a uložení protokolu.
    """
    if not os.path.exists(inp_file):
        raise FileNotFoundError(f"Vstupní soubor '{inp_file}' nebyl nalezen!")

    ref_density_m3 = (density_cm3 * 1.0e6) if density_cm3 is not None else None
    ref_bfield_t = (bfield_nt * 1.0e-9) if bfield_nt is not None else None

    converter = InpToSIConverter(inp_file, ref_density_m3=ref_density_m3, ref_bfield_tesla=ref_bfield_t)
    report_text = converter.generate_report()

    # Určení výstupního textového souboru
    if out_txt_file is None:
        base_name = os.path.splitext(os.path.basename(inp_file))[0]
        inp_dir = os.path.dirname(os.path.abspath(inp_file))
        out_txt_file = os.path.join(inp_dir, f"{base_name}_si_parameters.txt")

    with open(out_txt_file, 'w', encoding='utf-8') as f_out:
        f_out.write(report_text)

    return out_txt_file, report_text


def main():
    parser = argparse.ArgumentParser(
        description="Převodník vstupních parametrů iPIC3D (.inp) do jednotek SI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Příklady:
  python inp_to_si.py myfile.inp
  python inp_to_si.py myfile.inp -o parametry_si.txt
  python inp_to_si.py myfile.inp --density 10.0
  python inp_to_si.py ../../res_data_vth/beam_hot_tenuous_few/ele_beam_bump_on_tail_optimized.inp
"""
    )
    parser.add_argument("inp_file", nargs="?", default="myfile.inp", help="Cesta ke vstupnímu .inp souboru (výchozí: myfile.inp)")
    parser.add_argument("-o", "--output", default=None, help="Cesta k výstupnímu .txt souboru (výchozí: <vstup>_si_parameters.txt)")
    parser.add_argument("-d", "--density", type=float, default=None, help="Referenční hustota plazmatu n_0 v [cm^-3] (např. 10.0)")
    parser.add_argument("-b", "--bfield", type=float, default=None, help="Referenční magnetické pole |B| v [nT] (např. 10.0)")

    args = parser.parse_args()

    # Kontrola existence souboru
    target_inp = args.inp_file
    if not os.path.exists(target_inp):
        # zkusíme se podívat v aktuální složce skriptu
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, target_inp)
        if os.path.exists(alt_path):
            target_inp = alt_path
        else:
            print(f"CHYBA: Soubor '{args.inp_file}' nebyl nalezen!", file=sys.stderr)
            sys.exit(1)

    try:
        out_path, report = convert_inp_to_si(
            inp_file=target_inp,
            out_txt_file=args.output,
            density_cm3=args.density,
            bfield_nt=args.bfield
        )
        # Výpis do konzole
        print(report)
        print(f"\n[INFO] Protokol byl úspěšně uložen do souboru:\n  --> {out_path}\n")
    except Exception as e:
        print(f"CHYBA při zpracování: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
