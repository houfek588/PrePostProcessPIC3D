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
        self.time_const = time_const
        self.charge_const = charge_const

    def convert_velocity_SI_to_sim(self, vel_SI):
        return vel_SI / self.vel_const

    def convert_density_SI_to_sim(self, density_SI):
        return density_SI / self.rho_const

    def convert_time_SI_to_sim(self, time_SI):
        return time_SI * self.time_const

    def convert_length_SI_to_sim(self, len_SI):
        return len_SI / self.len_const

def convert_ev_to_kelvin(inp):
    return inp * (const_e/const_K_b)


# converts to PIC
def convert_rad_to_hz(inp):
    return (inp/(2*math.pi))




# input from Solar wind
# simulation
len_x = 1
nx = 4096
dt = 0.005


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
    print("\tplasma wave length = \t\t" + str(lambda_p_e) + " m", file=file)
    print(f"\tplasma frequency = \t\t\t{om_p_e} s-1 = {convert_rad_to_hz(om_p_e)} Hz", file=file)
    print(f"\tion frequency = \t\t\t{om_p_i} s-1 = {convert_rad_to_hz(om_p_i)} Hz", file=file)
    print(f"\tplasma wave period = \t\t{1/convert_rad_to_hz(om_p_e)} s", file=file)
    print(f"\tion wave period = \t\t\t{1/convert_rad_to_hz(om_p_i)} s", file=file)
    print("\tion skin depth = \t\t\t" + str(ion_skin_i) + " m", file=file)
    print("\telectron skin depth = \t\t" + str(ion_skin_e) + " m", file=file)


    print(f"\tion thermal velocity: \t\t{ion.get_thermal_velocity()} m/s", file=file)
    print(f"\telectron thermal velocity: \t{electron.get_thermal_velocity()} m/s", file=file)
    print(f"\tbeam velocity = \t\t\t{beam_velocity} m/s", file=file)
    print(str_line, file=file)

    print("SIMULATION PARAMETERS\n", file=file)
    print("\tmax time step: " + str(c1.convert_time_SI_to_sim(max_time_step * 0.1)), file=file)
    print("\tmax cell size: " + str(c1.convert_length_SI_to_sim(lambda_p_e) * 0.1), file=file)
    print("\tfor length: " + str(len_x) + " it is a least " + str(len_x/(c1.convert_length_SI_to_sim(lambda_p_e) * 0.1)) + " cells", file=file)


    dx = len_x/nx

    print(f"\nx length: {len_x} isd = {ion_skin_i} m;  (isd = ion skin depth)", file=file)
    print(f"dt time: {1} if = {1/om_p_i} s;  (if = ion frequency)", file=file)
    print(f"\tin SIM: x length: {len_x} isd = {len_x*ion_skin_i} m", file=file)
    print(f"\tin SIM: dt time: {dt} if = {(1 / om_p_i)*dt} s", file=file)
    print(f"number of cells in x direction is {nx}", file=file)
    print(f"\tin SIM: length step size: {(len_x*ion_skin_i)/nx} m", file=file)
    print(f"simulation resolution is dx = {dx} (should be < 1)", file=file)
    print(f"there should be {dx} < {ion_skin_e} < {ion_skin_i}", file=file)


    print("\nbackground proton parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.convert_velocity_SI_to_sim(ion.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0) + " ???", file=file)
    print("\tdensity: \t\t\t" + str(c1.convert_density_SI_to_sim(ion.get_concetration())), file=file)

    print("\nbackground electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.convert_velocity_SI_to_sim(electron.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0) + " ???", file=file)
    print("\tdensity: \t\t\t" + str(c1.convert_density_SI_to_sim(electron.get_concetration())), file=file)

    print("\nbeam electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.convert_velocity_SI_to_sim(electron_beam.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0) + " ???", file=file)
    print("\tdensity: \t\t\t" + str(c1.convert_density_SI_to_sim(electron_beam.get_concetration())), file=file)
    print("\tinject velocity: \t" + str(c1.convert_velocity_SI_to_sim(electron_beam.get_thermal_velocity() * 5)), file=file)

    print(str_line, file=file)

# print(f"name {__name__}")
if __name__ == '__main__':
    electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
    electron_beam = particles_parameters(const_e, T_b, const_M_e, n_b * 10 ** 6)
    ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)

    # print(electron)
    # print(ion)

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




    with open("../iPIC_par.txt", "w") as file:
        print_results(file)

    print_results(None)

    dt = 0.005
    real_T = dt/om_p_i

    print(f"om_p_i = {om_p_i}")
    print(f"om_p_i = {1/om_p_i} = 1 time unit")
    print(f"real dT = {(1/om_p_i) * dt}")
    print(f"dt_sim -> real : {dt * (1/om_p_i)} s")
    print(f"t/om_i : {4163 / om_p_i}")

    kk = (len_x / nx) * math.sqrt(64)
    print(kk)

