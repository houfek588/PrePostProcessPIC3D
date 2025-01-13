import math

# constants
const_K_b = 1.380649*10**(-23)
const_e = 1.602176634*10**(-19)
# const_M_e = 9.1093837139*10**(-31)
const_M_pr = 1.67262192595*10**(-27)
const_eps_0 = 8.8541878128*10**(-12)
const_c = 2.99792458*10**8
const_M_e = const_M_pr/64
# const_M_pr = 64*const_M_e

class Converter:
    def __init__(self, vel_const, len_const, rho_const, time_const, charge_const):
        self.vel_const = vel_const
        self.len_const = len_const
        self.rho_const = rho_const
        self.time_const = 1/time_const
        self.charge_const = charge_const

    def velocity_SI_to_sim(self, vel_SI):
        return vel_SI / self.vel_const

    def velocity_sim_to_SI(self, vel_SIM):
        return vel_SIM * self.vel_const

    def density_SI_to_sim(self, density_SI):
        return density_SI / self.rho_const

    def time_SI_to_sim(self, time_SI):
        return time_SI / self.time_const

    def time_sim_to_SI(self, time_SIM):
        return time_SIM * self.time_const

    def length_SI_to_sim(self, len_SI):
        return len_SI / self.len_const

    def length_sim_to_SI(self, len_SIM):
        return len_SIM * self.len_const

def convert_ev_to_kelvin(inp):
    return inp * (const_e/const_K_b)


# converts to PIC
def convert_rad_to_hz(inp):
    return (inp/(2*math.pi))


def get_debey_length(eps_0, K_B, temp, n_e, q_e):
    return math.sqrt((eps_0 * K_B * temp)/(n_e**6 * q_e**2))


# input from Solar wind
# simulation
len_x = 0.75
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
n_b = 0.001 * n_i   # cm-3
T_b = T_i   # eV
# v_b = 5*v_th

class particles_parameters:
    def __init__(self, charge, temp, mass, conc):
        self.charge = charge
        self.temp = temp
        self.mass = mass
        self.conc = conc

    def get_temp_in_ev(self):
        return self.temp

    def get_concetration(self):
        return self.conc

    def get_temp_in_kelvin(self):
        return self.temp * (const_e / const_K_b)

    def get_thermal_velocity(self):
        return math.sqrt((const_K_b * self.get_temp_in_kelvin()) / self.mass)

    def get_plasma_frequency(self):
        return math.sqrt((self.conc * self.charge ** 2) / (const_eps_0 * self.mass))


    def get_ion_skin_depth(self):
        return const_c / self.get_plasma_frequency()

    def __str__(self):
        print("------------")
        print("Particle parameters")
        print("charge: " + str(self.charge) + " C")
        print("mass: " + str(self.mass) + " kg")
        print("concentration: " + str(self.conc) + " m-3 = " + str(self.conc*10**-6) + " cm-3")
        print("temperature:")
        print("\tT_eV = " + str(self.get_temp_in_ev()) + " eV")
        print("\tT_k = " + str(self.get_temp_in_kelvin()) + " K")

        print("thermal velocity: " + str(self.get_thermal_velocity()) + " m/s")

        print("plasma frequency:")
        print("\tom_p = " + str(self.get_plasma_frequency()) + " s-1")
        print("\tom_p = " + str(convert_rad_to_hz(self.get_plasma_frequency())) + " Hz")

        print("ion skin depth: " + str(self.get_ion_skin_depth()) + " m")

        return "------------"

def print_results(file = None):
    str_line = "\n<--------------------------------------------------------------------------->\n"
    result_tab = "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"

    print(str_line, file=file)
    print("PHYSICAL INPUTS\n", file=file)
    print(f"\tmagnetic induction; \t\tB = {B} nT", file=file)
    print(f"\tion density; \t\t\t\tn_i = {n_i} cm-3", file=file)
    print(f"\tion temperature; \t\t\tT_i = {T_i} eV", file=file)
    print(f"\telectron density; \t\t\tn_i = {n_e} cm-3", file=file)
    print(f"\telectron temperature; \t\tT_e = {T_e} eV", file=file)

    print(f"\tbeam electron density; \t\tn_b = {n_b} cm-3", file=file)
    print(f"\tbeam electron temperature; \tT_b = {T_b} eV", file=file)
    print(f"\tbeam electron velocity; \tv_b = 5 * v_th", file=file)

    print(str_line, file=file)

    print("PHYSICAL PARAMETERS\n", file=file)
    print("\tdebye length = \t\t\t\t\t" + str(debye_len) + " m", file=file)
    print("\tplasma wave length = \t\t\t" + str(lambda_p_e) + " m", file=file)


    print(f"\n\telectron plasma frequency = \t{om_p_e} s-1 = {convert_rad_to_hz(om_p_e)} Hz", file=file)
    print(f"\telectron plasma wave period = \t{1/convert_rad_to_hz(om_p_e)} s", file=file)
    print("\telectron skin depth = \t\t\t" + str(ion_skin_e) + " m", file=file)
    print("\telectron temperature = \t\t\t" + str(electron.get_temp_in_kelvin()) + " K", file=file)

    print(f"\n\tion plasma frequency = \t\t\t{om_p_i} s-1 = {convert_rad_to_hz(om_p_i)} Hz", file=file)
    print(f"\tion plasma wave period = \t\t{1/convert_rad_to_hz(om_p_i)} s", file=file)
    print("\tion skin depth = \t\t\t\t" + str(ion_skin_i) + " m", file=file)
    print("\tion temperature = \t\t\t" + str(ion.get_temp_in_kelvin()) + " K", file=file)

    print(f"\tion thermal velocity: \t\t\t{ion.get_thermal_velocity()} m/s", file=file)
    print(f"\telectron thermal velocity: \t\t{electron.get_thermal_velocity()} m/s", file=file)
    print(f"\tbeam velocity = \t\t\t\t{beam_velocity} m/s", file=file)

    print("\nStability condition", file=file)
    print("\tc*dT <= dx", file=file)
    print(f"\t {const_c * sim_time_step} <= {sim_cell_length}", file=file)
    if const_c * sim_time_step <= sim_cell_length:
        print(f"{result_tab} Condition OK", file=file)
    else:
        print(f"{result_tab} Condition NOT OK !!!!", file=file)

    print("\n\tc = isk * om_i", file=file)
    print(f"\t {const_c} <= {ion_skin_i * om_p_i}", file=file)
    if const_c <= ion_skin_i * om_p_i:
        print(f"{result_tab} Condition OK", file=file)
    else:
        print(f"{result_tab} Condition NOT OK !!!!", file=file)

    print(f"duration of simulation: {c1.time_sim_to_SI(dt) * num_cycles * 1000} ms", file=file)

    print(str_line, file=file)







    print("SIMULATION PARAMETERS\n", file=file)
    print("\tmax time step: " + str(dx), file=file)
    print("\tmax cell size: " + str(c1.length_SI_to_sim(debye_len/1.5)), file=file)
    print("\tfor length: L = " + str(len_x) + " it is at least nc = " + str(round(len_x/(c1.length_SI_to_sim(debye_len/1.5)))) + " cells", file=file)

    print("\nNumerical condition", file=file)
    print("\tdT <= L/N", file=file)
    print(f"\t {dt} <= {dx}", file=file)

    if dt <= dx:
        print(f"{result_tab} Condition OK", file=file)
    else:
        print(f"{result_tab} Condition NOT OK !!!!", file=file)

    print(f"\nCheck light velocity, c/om_pi = {c1.velocity_SI_to_sim(const_c)}", file=file)
    # print(f"{const_c / om_p_i}  ----  {om_p_i}", file=file)
    # print(f"{const_c / c1.vel_const} ---- {c1.vel_const}", file=file)



    print(f"\nx length: {len_x} isd = {ion_skin_i} m;  (isd = ion skin depth)", file=file)
    print(f"dt time: {1} if = {1/om_p_i} s;  (if = ion frequency)", file=file)
    print(f"\tin SIM: x length: {len_x} isd = {len_x*ion_skin_i} m", file=file)
    print(f"\tin SIM: dt time: {dt} if = {sim_time_step} s", file=file)

    print(f"\nnumber of cells in x direction is {nx}", file=file)
    print(f"\tin SIM: length step size: {sim_cell_length} m", file=file)
    print(f"\tin SIM: length step size in Debye length: {debye_len/sim_cell_length} dl", file=file)

    if debye_len/sim_cell_length >= 1.5:
        print(f"{result_tab} Size step OK", file=file)
    else:
        print(f"{result_tab} Size step  too large !!!!", file=file)

    print(f"\nsimulation resolution is dx = {dx} (should be < 1)", file=file)
    print(f"there should be {dx} < {ion_skin_e} < {ion_skin_i}", file=file)
    print(f"\nlight length during 1 time step: {light_len} m, it travels through {light_len/sim_cell_length} cells", file=file)



    print("\nbackground proton parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(ion.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0), file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(ion.get_concetration())), file=file)

    print("\nbackground electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(electron.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0), file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(electron.get_concetration())), file=file)

    print("\nbeam electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(electron_beam.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(c1.velocity_SI_to_sim(beam_velocity)), file=file)
    # print("\tdrift velocity: \t" + str(0) + " ???", file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(electron_beam.get_concetration())), file=file)


    print(str_line, file=file)


electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
electron_beam = particles_parameters(const_e, T_b, const_M_e, n_b * 10 ** 6)
ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)
debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)

# print(f"name {__name__}")
if __name__ == '__main__':


    # print(electron)
    # print(ion)

    # debye_len = math.sqrt((const_eps_0 * const_K_b * electron.get_temp_in_kelvin())/(n_e**6 * const_e**2))

    om_p_e = electron.get_plasma_frequency()
    ion_skin_e = electron.get_ion_skin_depth()

    om_p_i = ion.get_plasma_frequency()
    ion_skin_i = ion.get_ion_skin_depth()

    beam_velocity = 5 * electron_beam.get_thermal_velocity()
    lambda_p_e = beam_velocity / convert_rad_to_hz(om_p_e)
    # print("plasma wave length = " + str(lambda_p_e) + " m, for beam velocity = " + str(beam_velocity) + " m/s")

    # convert to PIC values
    print(f"dt_e = {1 / convert_rad_to_hz(om_p_e)}")
    print(f"dt_i = {1 / convert_rad_to_hz(om_p_i)}")
    max_time_step = min([1 / convert_rad_to_hz(om_p_e), 1 / convert_rad_to_hz(om_p_i)])
    c1 = Converter(const_c, ion_skin_i, 1, om_p_i, const_e)

    print(f"max_time_step = {max_time_step}")
    # So the distribution of the absolute value of velocity (u) should follow the Maxwellian:
    # F = 4.*pi * (0.5/pi/uth**2)**(1.5) * u**2 * exp(-0.5*u**2/uth**2) * Ns * du,
    # where uth is thermal speed for the specie, Ns=npcelx*npcely*npcelz, du is velocity bin size.
    # Ns = 1024 * 1 * 128
    # uth = electron.get_thermal_velocity()
    # Ff = 4 * math.pi * (0.5/math.pi/uth**2)**(1.5) * u**2 * math.exp(-0.5*u**2/uth**2) * Ns * du

    dx = len_x / nx
    # sim_time_step = (1 / om_p_i) * dt
    # sim_cell_length = (len_x * ion_skin_i) / nx
    sim_time_step = c1.time_sim_to_SI(dt)
    sim_cell_length = c1.length_sim_to_SI(len_x/nx)

    light_len = const_c * sim_time_step




    with open("../iPIC_par.txt", "w") as file:
        print_results(file)

    print_results(None)

    # dt = 0.005
    # real_T = dt/om_p_i
    #
    print(f"om_ip = {om_p_i}")
    print(f"dt = {dt}")
    period_p_i = 1/om_p_i
    print(f"t_SI = {om_p_i}s  ->   t_sim = {c1.time_SI_to_sim(period_p_i)}")
    print(f"t_sim = 1     ->      t_SI = {1/c1.time_sim_to_SI(1)}")
    print(f"t_sim = {dt}     ->      t_SI = {c1.time_sim_to_SI(dt)}")

    nn = n_e
    mm = 1/nn
    mm_sim = c1.length_SI_to_sim(mm)
    print(f"\nn = {nn} m-1 -> m = 1/n = {mm} m")
    print(f"convert m to SIM unit -> m_SIM = {mm_sim} isd")
    print(f"value 1/m_SIM = {1/mm_sim} isd-1")

