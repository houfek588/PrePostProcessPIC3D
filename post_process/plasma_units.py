"""
plasma_units.py - Fyzikální jednotky, normalizace a převodní faktory pro iPIC3D postprocessing.

Tento modul zajišťuje úplnou soběstačnost balíčku post_process bez závislosti
na podsložce pre_process. Umožňuje používat výchozí škály, nebo načíst
přesné parametry přímo ze simulačního .inp souboru.
"""

import math
import os
import re
from typing import Optional, Dict, Any

# Základní fyzikální konstanty v SI
CONST_K_B = 1.380649e-23        # Boltzmannova konstanta [J/K]
CONST_E = 1.602176634e-19       # Elementární náboj [C]
CONST_M_E_REAL = 9.1093837139e-31  # Reálná hmotnost elektronu [kg]
CONST_M_PR = 1.67262192595e-27  # Hmotnost protonu [kg]
CONST_EPS_0 = 8.8541878128e-12  # Permitivita vakua [F/m]
CONST_C = 2.99792458e8          # Rychlost světla ve vakuu [m/s]
CONST_MU_0 = 4.0 * math.pi * 1e-7  # Permeabilita vakua [N/A^2]


class ParticleParameters:
    """Parametry daného druhu částic a odvozené fyzikální charakteristiky."""

    def __init__(self, charge: float, temp_ev: float, mass: float, conc_m3: float):
        self.charge = charge
        self.temp = temp_ev
        self.mass = mass
        self.conc = conc_m3

    def get_temp_in_ev(self) -> float:
        return self.temp

    def get_temp_in_kelvin(self) -> float:
        return self.temp * (CONST_E / CONST_K_B)

    def get_concentration(self, nx=None, Lx=None) -> float:
        return self.conc

    get_concetration = get_concentration  # Zpětná kompatibilita

    def get_thermal_velocity(self) -> float:
        """Tepelná rychlost v_th = sqrt(k_B * T / m) [m/s]."""
        return math.sqrt((CONST_K_B * self.get_temp_in_kelvin()) / self.mass)

    def get_plasma_frequency(self) -> float:
        """Plazmová frekvence omega_p = sqrt(n * q^2 / (eps_0 * m)) [rad/s]."""
        if self.mass <= 0 or self.conc <= 0:
            return 0.0
        return math.sqrt((self.conc * self.charge**2) / (CONST_EPS_0 * self.mass))

    def get_ion_skin_depth(self) -> float:
        """Skin depth d = c / omega_p [m]."""
        om_p = self.get_plasma_frequency()
        return CONST_C / om_p if om_p > 0 else 0.0


class UnitConverter:
    """Převodní konstanty mezi simulačními jednotkami iPIC3D a soustavou SI."""

    def __init__(self, vel_const: float, len_const: float, rho_const: float,
                 time_const: float, charge_const: float, mass_const: float):
        self.vel_const = vel_const
        self.len_const = len_const
        self.rho_const = rho_const
        self.time_const = 1.0 / time_const if time_const != 0 else 1.0
        self.charge_const = charge_const
        self.mass_const = mass_const

        self.energy_const = self.mass_const * (self.vel_const ** 2)
        self.e_field_const = (self.mass_const * self.vel_const) / (self.charge_const * self.time_const) if (self.charge_const * self.time_const) != 0 else 1.0
        self.b_field_const = self.mass_const / (self.charge_const * self.time_const) if (self.charge_const * self.time_const) != 0 else 1.0


class PlasmaUnits:
    """
    Zapouzdření normalizačních a fyzikálních parametrů plazmatu pro postprocessing.
    """

    def __init__(self, B_nT: float = 10.0, n_i_cm3: float = 10.0, T_i_eV: float = 10.0,
                 n_e_cm3: float = 10.0, T_e_eV: float = 10.0, mass_ratio: float = 64.0,
                 n_b_cm3: Optional[float] = None, T_b_eV: Optional[float] = None):
        self.const_c = CONST_C
        self.B_SI = B_nT * 1e-9
        self.mass_ratio = mass_ratio
        self.m_p = CONST_M_PR
        self.m_e = self.m_p / mass_ratio

        n_b = n_b_cm3 if n_b_cm3 is not None else (0.001 * n_e_cm3)
        T_b = T_b_eV if T_b_eV is not None else (0.3 * T_e_eV)

        self.ion = ParticleParameters(CONST_E, T_i_eV, self.m_p, n_i_cm3 * 1e6)
        self.electron = ParticleParameters(CONST_E, T_e_eV, self.m_e, n_e_cm3 * 1e6)
        self.electron_beam = ParticleParameters(CONST_E, T_b, self.m_e, n_b * 1e6)

        # Debyeova délka lambda_D = sqrt(eps_0 * k_B * T_e / (n_e * e^2))
        T_e_K = self.electron.get_temp_in_kelvin()
        if self.electron.conc > 0:
            self.debye_len = math.sqrt((CONST_EPS_0 * CONST_K_B * T_e_K) / (self.electron.conc * (CONST_E**2)))
        else:
            self.debye_len = 1.0

        om_p_i = self.ion.get_plasma_frequency()
        ion_skin_i = self.ion.get_ion_skin_depth()

        self.c1 = UnitConverter(
            vel_const=CONST_C,
            len_const=ion_skin_i,
            rho_const=self.ion.conc,
            time_const=om_p_i,
            charge_const=CONST_E,
            mass_const=self.m_p
        )

    @classmethod
    def from_inp_file(cls, inp_path: str) -> "PlasmaUnits":
        """
        Vytvoří instanci PlasmaUnits načtením parametrů přímo ze zadaného .inp souboru.
        """
        if not os.path.exists(inp_path):
            raise FileNotFoundError(f"Soubor .inp '{inp_path}' nebyl nalezen.")

        raw_text = ""
        with open(inp_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()

        def get_val(key: str, default: float = 0.0) -> float:
            m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^\n#;]+)", raw_text, re.MULTILINE | re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1).strip().split()[0])
                except ValueError:
                    pass
            return default

        B0x = get_val("B0x", 0.0)
        B0y = get_val("B0y", 0.0)
        B0z = get_val("B0z", 0.0)
        B_sim = math.sqrt(B0x**2 + B0y**2 + B0z**2)

        qom0 = abs(get_val("qom0", -64.0))
        qom1 = abs(get_val("qom1", 1.0))
        mass_ratio = qom0 / qom1 if qom1 != 0 else 64.0

        rhoINIT0 = get_val("rhoINIT0", 1.0)
        rhoINIT1 = get_val("rhoINIT1", 1.0)
        rhoINIT2 = get_val("rhoINIT2", 0.0)

        uth0 = get_val("uth0", 0.001)
        uth1 = get_val("uth1", uth0 / math.sqrt(mass_ratio) if mass_ratio > 0 else 0.001)
        uth2 = get_val("uth2", uth0)

        # Základní referenční hustota n_ref = 10 cm^-3
        n_ref_cm3 = 10.0
        n_i_cm3 = rhoINIT1 * n_ref_cm3
        n_e_cm3 = rhoINIT0 * n_ref_cm3
        n_b_cm3 = rhoINIT2 * n_ref_cm3 if rhoINIT2 > 0 else None

        m_p = CONST_M_PR
        m_e = m_p / mass_ratio

        # T [eV] = (m * (uth * c)^2) / e
        T_e_eV = (m_e * ((uth0 * CONST_C) ** 2)) / CONST_E if uth0 > 0 else 10.0
        T_i_eV = (m_p * ((uth1 * CONST_C) ** 2)) / CONST_E if uth1 > 0 else 10.0
        T_b_eV = (m_e * ((uth2 * CONST_C) ** 2)) / CONST_E if uth2 > 0 else 3.0

        # B_SI z B_sim
        om_pi_ref = math.sqrt((n_i_cm3 * 1e6 * (CONST_E**2)) / (CONST_EPS_0 * m_p))
        B_SI_ref = B_sim * (m_p * om_pi_ref / CONST_E)
        B_nT = B_SI_ref * 1e9 if B_SI_ref > 0 else 10.0

        return cls(
            B_nT=B_nT,
            n_i_cm3=n_i_cm3,
            T_i_eV=T_i_eV,
            n_e_cm3=n_e_cm3,
            T_e_eV=T_e_eV,
            mass_ratio=mass_ratio,
            n_b_cm3=n_b_cm3,
            T_b_eV=T_b_eV
        )


# Globální instance pro postprocessing
_GLOBAL_UNITS: Optional[PlasmaUnits] = None


def get_default_units() -> PlasmaUnits:
    """Vrátí globální instanci PlasmaUnits (líná inicializace)."""
    global _GLOBAL_UNITS
    if _GLOBAL_UNITS is None:
        _GLOBAL_UNITS = PlasmaUnits()
    return _GLOBAL_UNITS


def set_global_units(units: PlasmaUnits) -> None:
    """Nastaví globální instanci PlasmaUnits."""
    global _GLOBAL_UNITS
    _GLOBAL_UNITS = units
