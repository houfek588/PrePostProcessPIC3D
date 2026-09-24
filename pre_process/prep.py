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
debye_len = unit_input.get_debye_length(unit_input.const_eps_0, unit_input.const_K_b, electron.get_temp_in_kelvin(), electron.conc, unit_input.const_e)
om_p_i = ion.get_plasma_frequency()
ion_skin_i = ion.get_ion_skin_depth()

const_box = unit_input.ConstBox()
const_box.vel_const = unit_input.const_c
const_box.len_const = ion_skin_i
const_box.rho_const = ion.conc  # background ion density in m^-3
const_box.time_const = om_p_i
const_box.charge_const = unit_input.const_e
const_box.mass_const = unit_input.const_M_pr

c1 = unit_input.ConverterExt(const_box)

# Converted parameters
B_sim = c1.b_field_SI_to_sim(B * 10**(-9))

# Densities (normalized to ion density so total plasma is neutral: rho_e + rho_b = rho_i = 1.0)
rho_i = c1.density_SI_to_sim(ion.conc)            # 1.0
rho_b = c1.density_SI_to_sim(electron_beam.conc)  # 0.001
rho_e = rho_i - rho_b                             # 0.999

# Thermal velocities (normalized to c)
vth_e = c1.velocity_SI_to_sim(electron.get_thermal_velocity())
vth_i = c1.velocity_SI_to_sim(ion.get_thermal_velocity())
vth_b = c1.velocity_SI_to_sim(electron_beam.get_thermal_velocity())

# Drift velocity for beam in X direction
beam_velocity = 5 * electron_beam.get_thermal_velocity()
u0_b = c1.velocity_SI_to_sim(beam_velocity)

f = gen_inp_file.PIC3DInputFile("myfile.inp", "1Dwaves", 3)
f.set_case("none")
f.set_time_par(dt, num_cycles)
f.set_mag_field(B_sim, 0, 0)
f.set_box_size(len_x, 0.1, 0.1)
f.set_num_cells(nx, 1, 1)
f.set_mpi_topology(32, 1, 1)

# background electrons
f.set_part_par(0, -64, 0.0, rho_e)
f.set_thermal_vel(0, vth_e, vth_e, vth_e)
f.set_drift_vel(0, 0.0, 0.0, 0.0)
f.set_npcell_vel(0, 50, 1, 1)

# background ions
f.set_part_par(1, 1, 0.0, rho_i)
f.set_thermal_vel(1, vth_i, vth_i, vth_i)
f.set_drift_vel(1, 0.0, 0.0, 0.0)
f.set_npcell_vel(1, 50, 1, 1)

# beam electron
f.set_part_par(2, -64, 0.0, rho_b)
f.set_thermal_vel(2, vth_b, vth_b, vth_b)
f.set_drift_vel(2, u0_b, 0.0, 0.0)
f.set_npcell_vel(2, 30, 1, 1)

with open(f.get_file_name(), "w") as file:
    f.get_file_text(file)

print(f"Generated input file: {f.get_file_name()}")