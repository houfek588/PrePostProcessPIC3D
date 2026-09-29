"""
read_files.py - Moduly pro čtení simulačních dat iPIC3D (VTK, HDF5, textové výstupy).
"""

import re
import csv
import numpy as np
import pyvista as pv
import h5py
import unit_convert


def get_data_row(data, column, row):
    return data[column * row: row * (column + 1) - 1]


def max_value(inputlist):
    if isinstance(inputlist, np.ndarray):
        return float(np.nanmax(inputlist))
    return max(max(sublist) for sublist in inputlist)


def min_value(inputlist):
    if isinstance(inputlist, np.ndarray):
        return float(np.nanmin(inputlist))
    return min(min(sublist) for sublist in inputlist)


def add_suffix(file_name, suffix):
    splited_name = file_name.split(".")
    if len(splited_name) >= 2:
        return splited_name[0] + suffix + splited_name[1]
    return file_name + suffix


def avg_2n2_matrix(input_matrix):
    """Vypočítá průměr matice (podporuje jak NumPy, tak vnořené seznamy)."""
    if isinstance(input_matrix, np.ndarray):
        return float(np.mean(input_matrix))
    total_sum = sum(sum(row) for row in input_matrix)
    total_count = sum(len(row) for row in input_matrix)
    return total_sum / total_count if total_count > 0 else 0.0


def get_vector_variables():
    return ["E", "B"]


def parse_text_and_number(input_string):
    """
    Parsuje řetězec na textovou část a číselný index (např. 'cycle_100' -> ('cycle', 100)).
    """
    pattern = r"^(cycle)_(\d+)$"
    match = re.match(pattern, input_string)
    if match:
        return match.group(1), int(match.group(2))
    # Obecnější pattern pro formáty jako 'step123'
    gen_pattern = r"^([a-zA-Z_]+)(\d+)$"
    m = re.match(gen_pattern, input_string)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


# ---------------------------------------------------------------------------------
# BÁZOVÁ TŘÍDA PRO ČTENÍ DAT
# ---------------------------------------------------------------------------------

class BaseDataReader:
    """Společná bázová třída pro čtečky simulačních dat iPIC3D."""

    def __init__(self, variable="E", axis="x"):
        self.variable = variable
        self.axis = axis
        self.data_x_t = None
        self.len_data = None
        self.time_data = None

    def is_vector(self) -> bool:
        return self.variable in get_vector_variables()

    def get_field1D_len(self, t):
        if self.data_x_t is None:
            return []
        return self.data_x_t[t]

    def get_field1D_time(self, n):
        if self.data_x_t is None:
            return []
        if isinstance(self.data_x_t, np.ndarray):
            return self.data_x_t[:, n]
        return [float(d[n]) for d in self.data_x_t]

    def get_2D_data(self):
        return self.data_x_t

    def get_data_name(self):
        return self.variable

    def get_time_data(self):
        return self.time_data

    def get_len_data(self):
        return self.len_data


# ---------------------------------------------------------------------------------
# ČTEČKA VTK SOUBORŮ
# ---------------------------------------------------------------------------------

class ReadVTKFilesData(BaseDataReader):
    """Načítá časoprostorová data ze sady .vtk souborů (pole E, B, hustoty rho)."""

    def __init__(self, folder, file_for_graph, num_of_files, step, dt, nx, nz, variable="E", axis="x"):
        super().__init__(variable=variable, axis=axis)
        self.folder = folder
        self.num_of_files = num_of_files
        self.step = step
        self.dt = dt

        if isinstance(file_for_graph, list):
            self.file_paths = [folder + f if not f.startswith(folder) else f for f in file_for_graph]
        else:
            self.file_paths = self.create_file_names(folder, file_for_graph, num_of_files, step)

        if not self.is_vector():
            raw_data = self.read_file_time_dataset(self.file_paths, nx, nz)
            self.data_x_t = np.asarray(raw_data)
        else:
            raw_data = self.read_file_time_dataset(self.file_paths, nx, nz)
            self.data_x_t = self.preprocess_2D_data(raw_data, axis)

        print("All VTK files loaded...")

    def create_file_names(self, folder, file, num_of_files, step):
        pattern = r"^(.*)_(\d+)\.(\w+)$"
        match = re.match(pattern, file)
        if not match:
            raise ValueError("The file name does not match expected format: 'name_number.extension'")

        base_name, _, file_extension = match.groups()
        base_name_with_folder = f"{folder}{base_name}_"
        return [f"{base_name_with_folder}{i}.{file_extension}" for i in range(0, num_of_files, step)]

    def read_file_time_dataset(self, file_paths, grid_x, grid_z):
        if not isinstance(file_paths, list):
            raise ValueError("file_paths must be a list of strings representing file paths.")

        grow_data = []
        for path in file_paths:
            try:
                mesh = pv.read(path)
                data = mesh.point_data[mesh.array_names[0]]
                row_data = get_data_row(data, int(grid_z / 2), int(grid_x))
                grow_data.append(row_data)
            except FileNotFoundError:
                raise FileNotFoundError(f"File not found: {path}")
            except Exception as e:
                raise RuntimeError(f"Error while reading file: {path}. Error: {e}")

        return grow_data

    def preprocess_2D_data(self, raw_data, vector_component="x"):
        dim_mapping = {"x": 0, "y": 1, "z": 2}
        if vector_component not in dim_mapping:
            raise ValueError(f"Invalid vector component '{vector_component}'. Expected one of {list(dim_mapping.keys())}.")

        dim_idx = dim_mapping[vector_component]
        raw_arr = np.asarray(raw_data)
        if raw_arr.ndim == 3:
            return raw_arr[:, :, dim_idx]
        return np.array([[float(d[dim_idx]) for d in time_step] for time_step in raw_data])


# ---------------------------------------------------------------------------------
# ČTEČKA HDF5 POLÍ
# ---------------------------------------------------------------------------------

class ReadHDFFieldData(BaseDataReader):
    """Načítá pole z HDF5 souborů generovaných iPIC3D."""

    def __init__(self, folder, file_for_graph, num_of_files, variable="E", axis="x"):
        super().__init__(variable=variable, axis=axis)
        self.folder = folder
        self.num_of_files = num_of_files
        self.file_paths = self.create_file_names(folder, file_for_graph, num_of_files)
        self.data_x_t = self.read_field_2D()
        print("All HDF field files loaded...")

    def create_file_names(self, folder, file, num_of_files):
        split_names = file.split(".")
        name = split_names[0]
        ext = split_names[1] if len(split_names) > 1 else "hdf"

        pattern = r"([a-zA-Z_]+)(\d{1,3})$"
        match = re.match(pattern, name)
        if match:
            text_part = match.group(1)
        else:
            text_part = "proc"

        return [f"{folder}{text_part}{i}.{ext}" for i in range(0, num_of_files)]

    def read_field1D_len(self, hdf_files, var, cycle_key):
        var_data = []
        for file_path in hdf_files:
            with h5py.File(file_path, "r") as hdf:
                if "fields" not in hdf or var not in hdf["fields"] or cycle_key not in hdf["fields"][var]:
                    raise KeyError(f"Missing required keys in file {file_path}")
                cycle_data = hdf["fields"][var][cycle_key]
                for c in cycle_data:
                    var_data.append(avg_2n2_matrix(c))
                if var_data:
                    var_data.pop(-1)

        if self.len_data is None:
            self.len_data = list(range(0, len(var_data)))

        return var_data

    def read_field_2D(self):
        col_name = (self.variable + self.axis) if self.is_vector() else self.variable

        with h5py.File(self.file_paths[0], "r") as hdf:
            keys = list(hdf["fields"][col_name].keys())

        sort_cycle_name = sorted(keys, key=lambda x: parse_text_and_number(x)[1] if parse_text_and_number(x)[1] is not None else 0)

        if self.time_data is None:
            self.create_time_data(sort_cycle_name)

        data = [self.read_field1D_len(self.file_paths, col_name, c) for c in sort_cycle_name]
        return np.asarray(data)

    def create_time_data(self, sorted_cycles):
        self.time_data = [parse_text_and_number(i)[1] for i in sorted_cycles if parse_text_and_number(i)[1] is not None]


# ---------------------------------------------------------------------------------
# ČTEČKA HDF5 ČÁSTIC
# ---------------------------------------------------------------------------------

class ReadHDFParticleData(ReadHDFFieldData):
    """Načítá částicová data z HDF5 souborů generovaných iPIC3D."""

    def read_field_2D(self):
        species = self.variable
        var = self.axis

        with h5py.File(self.file_paths[0], "r") as hdf:
            keys = list(hdf["particles"][species][var].keys())

        sort_cycle_name = sorted(keys, key=lambda x: parse_text_and_number(x)[1] if parse_text_and_number(x)[1] is not None else 0)

        if self.time_data is None:
            self.create_time_data(sort_cycle_name)

        data = [self.read_field1D_len(self.file_paths, species, var, c) for c in sort_cycle_name]
        return np.asarray(data)

    def read_field1D_len(self, hdf_files, species, var, cycle_key):
        var_data = []
        for file_path in hdf_files:
            with h5py.File(file_path, "r") as hdf:
                if "particles" not in hdf or species not in hdf["particles"] or var not in hdf["particles"][species]:
                    continue
                if cycle_key not in hdf["particles"][species][var]:
                    continue

                cycle_data = hdf["particles"][species][var][cycle_key]
                for c in cycle_data:
                    var_data.append(c)

        try:
            sett = ReadHDFSettings(self.folder + "settings.hdf")
            part_in_cel = sett.get_num_part_in_cell(self.variable, self.axis)
            if self.len_data is None and part_in_cel > 0:
                raw = list(range(0, len(var_data)))
                self.len_data = unit_convert.rescale_list(raw, 1.0 / part_in_cel)
        except Exception:
            if self.len_data is None:
                self.len_data = list(range(0, len(var_data)))

        return var_data

    def get_particle1D_len(self, time):
        return self.get_field1D_len(time)

    def get_particle1D_time(self, length):
        return self.get_field1D_time(length)


# ---------------------------------------------------------------------------------
# ČTEČKA NASTAVENÍ (settings.hdf)
# ---------------------------------------------------------------------------------

class ReadHDFSettings:
    """Načítá simulační metadata z settings.hdf s cachováním v paměti."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.dict_dir = {"x": "x", "y": "y", "z": "z", "u": "x", "v": "y", "w": "z", "q": "x"}
        self.collective = {}
        self.load_settings()

    def load_settings(self):
        try:
            with h5py.File(self.file_name, "r") as hdf:
                if "collective" in hdf:
                    for k in hdf["collective"].keys():
                        item = hdf["collective"][k]
                        if isinstance(item, h5py.Dataset):
                            val = item[()]
                            if isinstance(val, np.ndarray) and val.size == 1:
                                self.collective[k] = val.item()
                            else:
                                self.collective[k] = val
                        elif isinstance(item, h5py.Group):
                            sub_dict = {}
                            for sub_k in item.keys():
                                sub_val = item[sub_k][()]
                                if isinstance(sub_val, np.ndarray) and sub_val.size == 1:
                                    sub_dict[sub_k] = sub_val.item()
                                else:
                                    sub_dict[sub_k] = sub_val
                            self.collective[k] = sub_dict
        except Exception as e:
            # Fallback pokud soubor neexistuje při offline testech
            print(f"[Warning] Could not load '{self.file_name}': {e}")

    def check_available_dir(self, direction):
        available = list(self.dict_dir.keys())
        if direction not in available:
            raise ValueError(f"Invalid direction '{direction}'. Must be one of {available}.")
        return True

    def get_num_cells(self, direction="x"):
        key = f"N{direction}c"
        if key in self.collective:
            return int(self.collective[key])
        return 4096 if direction == "x" else 1

    def get_box_size(self, direction="x"):
        self.check_available_dir(direction)
        key = f"L{self.dict_dir[direction]}"
        if key in self.collective:
            return float(self.collective[key])
        return 0.5 if direction == "x" else 0.1

    def get_num_cycles(self):
        return int(self.collective.get("Ncycles", 80000))

    def get_num_species(self):
        return int(self.collective.get("Ns", 3))

    def get_num_part_in_cell(self, species, direction="x"):
        self.check_available_dir(direction)
        cell_key = f"Npcel{self.dict_dir[direction]}"
        if species in self.collective and isinstance(self.collective[species], dict):
            return int(self.collective[species].get(cell_key, 50))
        return 50

    def get_time_step_size(self):
        return float(self.collective.get("Dt", 0.0001))

    def get_time_step_cycles(self):
        return int(self.collective.get("Ncycles", 80000))


# ---------------------------------------------------------------------------------
# ČTEČKA KONZERVOVANÝCH VELIČIN (ConservedQuantities.txt)
# ---------------------------------------------------------------------------------

class ReadConsData:
    """Načítá a parsuje ConservedQuantities.txt pro kontrolu zachování energie a hybnosti."""

    def __init__(self, folder, file_for_graph):
        path = folder + file_for_graph if not file_for_graph.startswith(folder) else file_for_graph
        self.data = {
            "cycle": ["Cycle", []],
            "total_energy": ["Total Energy", []],
            "momentum": ["Momentum", []],
            "e_energy": ["Electric Field Energy", []],
            "b_energy": ["Magnetic Field Energy", []],
            "k_energy": ["Kinetic Energy", []],
            "k_energy_spec0": ["Kinetic Energy Species 0", []],
            "k_energy_spec1": ["Kinetic Energy Species 1", []],
            "k_energy_spec2": ["Kinetic Energy Species 2", []],
            "bulk_energy_spec0": ["Bulk Energy Species 0", []],
            "bulk_energy_spec1": ["Bulk Energy Species 1", []],
            "bulk_energy_spec2": ["Bulk Energy Species 2", []]
        }
        self.read_file(path)

    def read_file(self, file_path):
        keys = [
            "cycle",
            None,
            "total_energy",
            "momentum",
            "e_energy",
            "b_energy",
            "k_energy",
            "k_energy_spec0",
            "k_energy_spec1",
            "k_energy_spec2",
            "bulk_energy_spec0",
            "bulk_energy_spec1",
            "bulk_energy_spec2"
        ]

        try:
            with open(file_path, mode="r", encoding="utf-8") as file:
                reader = csv.reader(file, delimiter="\t")
                for i, row in enumerate(reader):
                    if not row:
                        continue
                    if i == 0:
                        continue  # Header
                    for idx, key in enumerate(keys):
                        if key and idx < len(row):
                            try:
                                self.data[key][1].append(float(row[idx]))
                            except ValueError:
                                pass
        except Exception as e:
            print(f"[Warning] Error reading '{file_path}': {e}")

    def get_cycles(self):
        return self.data["cycle"]

    def get_total_energy(self):
        return self.data["total_energy"]

    def get_e_energy(self):
        return self.data["e_energy"]

    def get_k_energy(self):
        return self.data["k_energy"]

    def get_k_spec0_energy(self):
        return self.data["k_energy_spec0"]

    def get_k_spec1_energy(self):
        return self.data["k_energy_spec1"]

    def get_k_spec2_energy(self):
        return self.data["k_energy_spec2"]


# ---------------------------------------------------------------------------------
# 2D FAST FOURIER TRANSFORM
# ---------------------------------------------------------------------------------

def fft_2d(E, X, Y):
    """
    Provede 2D FFT nad polem E(y, x) s odpovídajícími osami X a Y.
    """
    E_arr = np.asarray(E)
    Ny, Nx = E_arr.shape
    if Nx != len(X) or Ny != len(Y):
        print(f"Error: Dimensions of E ({Ny}, {Nx}) and X ({len(X)}), Y ({len(Y)}) are incompatible.")
        return None

    E_fft = np.fft.fft2(E_arr)
    E_fft_shifted = np.fft.fftshift(E_fft)

    dx = float(np.diff(X).mean()) if len(X) > 1 else 1.0
    dy = float(np.diff(Y).mean()) if len(Y) > 1 else 1.0

    kx = np.fft.fftfreq(Nx, d=dx) * (2.0 * np.pi)
    ky = np.fft.fftfreq(Ny, d=dy) * (2.0 * np.pi)

    kx_shifted = np.fft.fftshift(kx)
    ky_shifted = np.fft.fftshift(ky)

    return E_fft_shifted, kx_shifted, ky_shifted