import unit_input
import gen_inp_file

# input from Solar wind
# simulation
len_x = 0.5
nx = 4096
dt = 0.0001
num_cycles = 80000


# background
B = 10      # nT
n_i = 10    # cm-3
n_e = n_i   # cm-3
T_i = 10    # eV
T_e = T_i   # eV

# beam
n_b = 0.001 * n_e   # cm-3
T_b = T_e * 0.3   # eV
# v_b = 5*v_th


electron = unit_input.particles_parameters(unit_input.const_e, T_e, unit_input.const_M_e, n_e * 10 ** 6)
electron_beam = unit_input.particles_parameters(unit_input.const_e, T_b, unit_input.const_M_e, n_b * 10 ** 6)
ion = unit_input.particles_parameters(unit_input.const_e, T_i, unit_input.const_M_pr, n_i * 10 ** 6)
debye_len = unit_input.get_debey_length(unit_input.const_eps_0, unit_input.const_K_b, electron.get_temp_in_kelvin(), n_e, unit_input.const_e)
om_p_i = ion.get_plasma_frequency()
ion_skin_i = ion.get_ion_skin_depth()

const_box = unit_input.ConstBox()
const_box.vel_const = unit_input.const_c
const_box.len_const = ion_skin_i
const_box.rho_const = 1
const_box.time_const = om_p_i
const_box.charge_const = unit_input.const_e
const_box.mass_const = unit_input.const_M_pr

c1 = unit_input.ConverterExt(const_box)


# print(f"\tmag. field: {c1.b_field_SI_to_sim(B*10**(-9))}", file=file)

f = gen_inp_file.PIC3DInputFile("myfile.inp", "1Dwaves", 3)

f.set_time_par(0.0001, 80000)
f.set_mag_field(c1.b_field_SI_to_sim(B*10**(-9)), 0, 0)
f.set_box_size(0.5, 0.1, 0.1)
f.set_num_cells(4096, 1, 1)

# backgound electrons
f.set_part_par(0, -64, 0.0, 0.99)
f.set_thermal_vel(0, 0.0008, 0.0, 0.0008)
f.set_drift_vel(0, 0.0, 0.0, 0.0)
f.set_npcell_vel(0, 50, 1, 1)

# background ions
f.set_part_par(1, 1, 0.0, 1)
f.set_thermal_vel(1, 0.0001,0.0, 0.0001)
f.set_drift_vel(1, 0.0, 0.0, 0.0)
f.set_npcell_vel(1, 50, 1, 1)

# beam electron
f.set_part_par(2, -64, 0.0, 0.01)
f.set_thermal_vel(2, 0.0004, 0.0008, 0.0008)
f.set_drift_vel(2, 0.002, 0.0, 0.0)
f.set_npcell_vel(2, 30, 1, 1)

with open(f.get_file_name(), "w") as file:
    f.get_file_text(file)