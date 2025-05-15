import math
import matplotlib.pyplot as plt

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


class ConstBox:
    def __init__(self):
        self.vel_const = 0
        self.len_const = 0
        self.rho_const = 0
        self.time_const = 0
        self.charge_const = 0
        self.mass_const = 0

    def check_fill_const(self):
        pass


class ConverterExt(Converter):
    def __init__(self, const_box: ConstBox):
        super().__init__(const_box.vel_const, const_box.len_const, const_box.rho_const, const_box.time_const
                         , const_box.charge_const)

        self.mass_const = const_box.mass_const
        self.energy_const = self.mass_const * self.vel_const**2
        self.e_field_const = (self.mass_const * self.vel_const)/(self.charge_const * self.time_const)
        self.b_field_const = self.mass_const/(self.charge_const*self.time_const)


    def charge_SI_to_sim(self, charge_SI):
        print(f"charge: {charge_SI} / {self.charge_const}")
        return charge_SI / self.charge_const

    def charge_sim_to_SI(self, charge_SIM):
        return charge_SIM * self.charge_const

    def mass_SI_to_sim(self, mass_SI):
        return mass_SI / self.mass_const

    def mass_sim_to_SI(self, mass_SIM):
        return mass_SIM * self.mass_const

    def energy_SI_to_sim(self, energy_SI):
        return energy_SI / self.energy_const

    def energy_sim_to_SI(self, energy_SIM):
        return energy_SIM * self.energy_const

    def e_field_SI_to_sim(self, e_field_SI):
        return e_field_SI / self.e_field_const

    def e_field_sim_to_SI(self, e_field_SIM):
        return e_field_SIM * self.e_field_const

    def b_field_SI_to_sim(self, b_field_SI):
        return b_field_SI / self.b_field_const

    def b_field_sim_to_SI(self, b_field_SIM):
        return b_field_SIM * self.b_field_const


def convert_ev_to_kelvin(inp):
    return inp * (const_e/const_K_b)


# converts to PIC
def convert_rad_to_hz(inp):
    return (inp/(2*math.pi))


def get_debey_length(eps_0, K_B, temp, n_e, q_e):
    return math.sqrt((eps_0 * K_B * temp)/(n_e**6 * q_e**2))


class particles_parameters:
    def __init__(self, charge, temp, mass, conc):
        self.charge = charge
        self.temp = temp
        self.mass = mass
        self.conc = conc

    def get_temp_in_ev(self):
        return self.temp

    def get_concetration(self, nx, Lx):
        print(f"nx: {nx}")
        print(f"Lx: {Lx}")

        ax = pow(self.conc, 1/3)
        cell_size = Lx/nx

        part = cell_size*ax

        print(f"ax: {ax}")
        print(f"cell_size: {cell_size}")
        print(f"part: {part}")

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
    print(f"\tmagnetic field: \t\t\t\t{B*10**(-9)} T", file=file)


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
    print(f"light length during 1 time step: {light_len} m, it travels through {round(light_len/sim_cell_length,3)} cells < 1", file=file)
    if light_len/sim_cell_length <= 1:
        print(f"{result_tab} Condition OK", file=file)
    else:
        print(f"\n{result_tab} Condition NOT OK !!!!", file=file)

    print("Input magnetic field", file=file)
    print(f"\tmag. field: {c1.b_field_SI_to_sim(B*10**(-9))}", file=file)


    print("\nbackground proton parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(ion.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0), file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(ion.get_concetration(nx,ion_skin_i))), file=file)

    print("\nbackground electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(electron.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(0), file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(electron.get_concetration(nx,ion_skin_i))), file=file)

    print("\nbeam electron parameters:", file=file)
    print("\tthermal velocity: \t" + str(c1.velocity_SI_to_sim(electron_beam.get_thermal_velocity())), file=file)
    print("\tdrift velocity: \t" + str(c1.velocity_SI_to_sim(beam_velocity)), file=file)
    # print("\tdrift velocity: \t" + str(0) + " ???", file=file)
    print("\tdensity: \t\t\t" + str(c1.density_SI_to_sim(electron_beam.get_concetration(nx,ion_skin_i))), file=file)


    print(str_line, file=file)


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


electron = particles_parameters(const_e, T_e, const_M_e, n_e * 10 ** 6)
electron_beam = particles_parameters(const_e, T_b, const_M_e, n_b * 10 ** 6)
ion = particles_parameters(const_e, T_i, const_M_pr, n_i * 10 ** 6)
debye_len = get_debey_length(const_eps_0, const_K_b, electron.get_temp_in_kelvin(), n_e, const_e)
om_p_i = ion.get_plasma_frequency()
ion_skin_i = ion.get_ion_skin_depth()

const_box = ConstBox()
const_box.vel_const = const_c
const_box.len_const = ion_skin_i
const_box.rho_const = 1
const_box.time_const = om_p_i
const_box.charge_const = const_e
const_box.mass_const = const_M_pr

c1 = ConverterExt(const_box)

# print(f"name {__name__}")
if __name__ == '__main__':


    # print(electron)
    # print(ion)

    # debye_len = math.sqrt((const_eps_0 * const_K_b * electron.get_temp_in_kelvin())/(n_e**6 * const_e**2))

    om_p_e = electron.get_plasma_frequency()
    ion_skin_e = electron.get_ion_skin_depth()



    beam_velocity = 5 * electron_beam.get_thermal_velocity()
    lambda_p_e = beam_velocity / convert_rad_to_hz(om_p_e)
    # print("plasma wave length = " + str(lambda_p_e) + " m, for beam velocity = " + str(beam_velocity) + " m/s")

    # convert to PIC values
    print(f"dt_e = {1 / convert_rad_to_hz(om_p_e)}")
    print(f"dt_i = {1 / convert_rad_to_hz(om_p_i)}")
    max_time_step = min([1 / convert_rad_to_hz(om_p_e), 1 / convert_rad_to_hz(om_p_i)])
    # c1 = Converter(const_c, ion_skin_i, 1, om_p_i, const_e)

    const_box = ConstBox()
    const_box.vel_const = const_c
    const_box.len_const = ion_skin_i
    const_box.rho_const = 1
    const_box.time_const = om_p_i
    const_box.charge_const = const_e
    const_box.mass_const = const_M_pr

    c1 = ConverterExt(const_box)

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
    # print(f"om_ip = {om_p_i}")
    # print(f"dt = {dt}")
    # period_p_i = 1/om_p_i
    # print(f"t_SI = {om_p_i}s  ->   t_sim = {c1.time_SI_to_sim(period_p_i)}")
    # print(f"t_sim = 1     ->      t_SI = {1/c1.time_sim_to_SI(1)}")
    # print(f"t_sim = {dt}     ->      t_SI = {c1.time_sim_to_SI(dt)}")
    #
    # nn = n_e
    # mm = 1/nn
    # mm_sim = c1.length_SI_to_sim(mm)
    # print(f"\nn = {nn} m-1 -> m = 1/n = {mm} m")
    # print(f"convert m to SIM unit -> m_SIM = {mm_sim} isd")
    # print(f"value 1/m_SIM = {1/mm_sim} isd-1")
    #
    print("\nrecalc.")
    N = 300000
    T = N*dt
    t_si = c1.time_sim_to_SI(T)
    c2 = Converter(const_c, ion_skin_e, 1, om_p_e, const_e)
    print(f"t = {N} cycles")
    print(f"t = {T} 1/om_pi")
    print(f"t = {t_si} s")
    print(f"t = {c2.time_SI_to_sim(t_si)} 1/om_pe")

    # convert Joule, Tesla and V/m to simulation units
    # Ek = 0,5*m*v2
    const_box = ConstBox()
    const_box.vel_const = const_c
    const_box.len_const = ion_skin_i
    const_box.rho_const = 1
    const_box.time_const = om_p_i
    const_box.charge_const = const_e
    const_box.mass_const = const_M_pr

    c3 = ConverterExt(const_box)

    # joule
    m1 = const_M_pr
    v1 = const_c
    Ek = 0.5 * m1 * v1**2
    print(f"\n\nEk = {Ek}")
    v1s = c3.velocity_SI_to_sim(v1)
    m2s = c3.mass_SI_to_sim(m1)
    EkS = c3.energy_SI_to_sim(Ek)
    Ek1s = 0.5*m2s*v1s**2
    print(f"conv, mass: {m2s}, vel: {v1s}")
    print(f"Ek1: {EkS}, Ek2: {Ek1s}")


    print(f"\nconv back; E: {c3.energy_sim_to_SI(EkS)}, m: {c3.mass_sim_to_SI(m2s)}, v: {c3.velocity_sim_to_SI(v1s)}")

    # V/m
    m2 = const_M_pr
    v2 = const_c
    q2 = const_e
    t2 = om_p_i
    Efield = (m2*v2)/(q2*t2)

    m2s = c3.mass_SI_to_sim(m2)
    v2s = c3.velocity_SI_to_sim(v2)
    q2s = c3.charge_SI_to_sim(q2)
    t2s = c3.time_SI_to_sim(t2)
    print(f"check; t = {c3.time_SI_to_sim(t_si)} 1/om_pe (t = {t_si}, c1: {c1.time_SI_to_sim(t_si)})")
    Efields = (m2s * v2s) / (q2s * t2s)

    print(f"el. field: {Efield} J")
    print(f"el. field: {Efields}")
    print(f"E test: {Efield * (const_e/(const_M_pr*const_c*om_p_i))}")
    print(f"E test2: {c3.e_field_SI_to_sim(Efield)}")
    print(f"E test back: {c3.e_field_sim_to_SI(c3.e_field_SI_to_sim(Efield))}")
    print(f"E test back: {c3.e_field_sim_to_SI(1e-5)}")

    print(f"\nconv back; q: {c3.charge_sim_to_SI(q2s)} =? {q2}, t: {c3.time_sim_to_SI(t2s)} =? {t2}")
    print(f"\nTime const: c2: {c1.time_const}, c3: {c3.time_const}")

    m3 = const_M_pr
    v3 = const_c
    q3 = const_e
    t3 = om_p_i
    Bfield = m3 / (q3 * t3)

    m3s = c3.mass_SI_to_sim(m3)
    v3s = c3.velocity_SI_to_sim(v3)
    q3s = c3.charge_SI_to_sim(q3)
    t3s = c3.time_SI_to_sim(t3)

    Bfields = m3s / (q3s * t3s)

    print(f"\n\nmag. field: {Bfield} T")
    print(f"mag. field: {Bfields}")
    print(f"B test: {Bfield * (const_e / (const_M_pr * om_p_i))}")
    print(f"B test2: {c3.b_field_SI_to_sim(Bfield)}")
    print(f"B test back: {c3.e_field_sim_to_SI(c3.b_field_SI_to_sim(Bfield))}")
    print(f"B test back: {c3.b_field_SI_to_sim(B*10**(-9))}")

    print(f"\nconv back; q: {c3.charge_sim_to_SI(q2s)} =? {q2}, t: {c3.time_sim_to_SI(t2s)} =? {t2}")
    print(f"\nTime const: c2: {c1.time_const}, c3: {c3.time_const}")

    print(c1.rho_const)


    # damping
    T_el = electron.get_temp_in_kelvin()
    k = 1

    om2 = (om_p_e**2) + 3*(k**2)*(const_K_b*T_el/const_M_e)

    gama = -math.sqrt(math.pi/8)*(om_p_e/math.fabs(k*debye_len)**3)*math.exp((-1/(2*(k*debye_len)**2))-3/2)

    o = []
    kk = []
    gam = []
    for i in range(1, 100, 1):
        k = (i * 0.01)/10.0
        omm = (om_p_e**2) + 3*(k**2)*(const_K_b*T_el/const_M_e)

        print(f"k*lamD = {k * debye_len}, k: {k}, lam: {debye_len}")
        print(math.fabs(k * debye_len) ** 3)
        gg = -math.sqrt(math.pi / 8) * (om_p_e / (math.fabs(k * debye_len) ** 3)) * math.exp(
            (-1 / (2 * (k * debye_len) ** 2)) - 3 / 2)

        o.append(math.sqrt(omm) / om_p_e)
        kk.append(k * debye_len)
        gam.append(-gg / om_p_e)

    # d = plot.PlotDescription()
    # d.multidata_labels(["om", "gama"])
    # plot.plot_data(kk, [o,gam], d)
    #
    # plot.plot_all_graphs()
    scale = 2
    # fig, ax = plt.subplots(figsize=(16 / scale, 9 / scale))
    plt.loglog(kk, o, label="y = x^2", color="blue")
    plt.loglog(kk, gam,  label="y = x^1.5", color="orange")

        # plt.scatter(dataX, dataY, color='red', label="Data Points", zorder=3)

    # plt.xlabel(descr.label_x)
    # plt.ylabel(descr.label_y)
    # plt.title(descr.title)
    plt.ylim(10**-4, 5)
    plt.grid(True)
    plt.legend()
    plt.show()
