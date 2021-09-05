import math
import numpy as np
from scipy import integrate
from scipy.special import spherical_jn, spherical_yn

from utils import conversions

# PaperP for PhysRevA.45.202
# PaperJ for J. Phys. B: Atom. Mol. Phys

# The use of Bessel function from Shouyi's code.
def riccati_bessel_first(order, argument):
    '''
    Riccati bessel function of the first kind.

    Parameters
    ----------
    order : Int
        DESCRIPTION. A real or complex value, in this case it's passed
                     a real
    argument : Float
        DESCRIPTION. Value of the Riccatci bessel function of the first kind

    Returns
    -------
    TYPE : Float
        DESCRIPTION. Value of the Riccatci bessel function of the first kind

    '''

    return argument * spherical_jn(order, argument)


def riccati_bessel_second(order, argument):
    '''
    Riccati bessel function of the second kind.

    Parameters
    ----------
    order : Int
        DESCRIPTION. A real or complex value, in this case it's passed
                     a real
    argument : Float
        DESCRIPTION. Value of the Riccatci bessel function of the second kind

    Returns
    -------
    TYPE : Float
        DESCRIPTION. Value of the Riccatci bessel function of the second kind

    '''

    return argument * spherical_yn(order, argument)

# Define parameters of five atoms and molecules ()
class Atom(object):
    def __init__(self, electron_num, delta, IP, alpha_0, rs, charge_density_path):
        super().__init__()
        self.electron_num = electron_num
        self.charge_density_path = charge_density_path
        self.delta = delta
        self.IP = IP
        self.alpha_0 = alpha_0
        self.rs = rs
        self.energy = None
        self.order = None

        self.__rho_r = dict()
        self.r_list = []
        self.rho_list = []
        with open(self.charge_density_path) as f:
            for line in f.readlines():
                r, rho_r = line.strip().split(',')
                r = round(float(r), 5)
                rho_r = float(rho_r)
                self.r_list.append(r)
                self.rho_list.append(rho_r)
                self.__rho_r[r] = rho_r

        # For calcualting V_st
        self.r1_values = None
        self.y_values = None
# Define charge desities.
    def get_rho_r(self, radius):
        if np.all(np.diff(self.r_list) > 0) and \
                np.all(radius > min(self.r_list) \
                and np.all(radius < max(self.r_list))):
            new_y = np.interp(radius, self.r_list, self.rho_list)
        elif round(radius, 5) == self.r_list[0]:
            new_y = self.rho_list[0]
        elif round(radius, 5) == self.r_list[-1]:
            new_y = self.rho_list[-1]
        else:
            print(radius)
            print(self.r_list[0])
            print(self.r_list[-1])
            raise ValueError("radius is out of range")
        return new_y

    def update_r1_values(self):
        r_list = np.linspace(0.001, 0.001 * 5000, num=5000, endpoint=True)
        charge_list = []    
        for r in r_list:
            charge = self.get_rho_r(r)
            charge_list.append(charge)
        self.r1_values = r_list
        self.y_values = np.array(charge_list)
# modeling V_st (static contribution)
    def get_v_st(self, radius, r1_values=None, y_values=None, epsilon=1e-6):
        if r1_values is None or y_values is None:
            r1_values = self.r1_values
            y_values = self.y_values
        y_list = []
        r1_list = []
        for i in range(len(r1_values)):
            r1 = r1_values[i]
            rho_r1 = y_values[i]

            if abs(radius - r1) < epsilon:
                continue
            else:
                r1_list.append(r1)
                y = rho_r1 / abs(radius - r1)
                y_list.append(y)
        return integrate.trapz(y_list, r1_list) - self.electron_num / radius
# Define some parameters for modeling v_ex in jain's paper. (However, we finally choose the model for v_ex in Hara's paper)
# So in Hara's model, K_F is replaced by k_max. We also define the new parameter in the following.
    def _get_k_F_from_r(self, radius):
        return math.pow(3 * (np.pi ** 2) * self.get_rho_r(radius), 1./3)

    def _get_k_max(self, radius):
        return math.pow(3 * np.pi  * self.get_rho_r(radius), 1./3)

    def _get_eta_from_r(self, radius, energy):
        k_F_r = self._get_k_F_from_r(radius)
        return np.sqrt(energy ** 2 + 2 * self.IP + k_F_r ** 2) / k_F_r
# modeling V_ex (exchange term)
    def get_v_ex(self, radius, energy):
        # get V_ex from r
        # PaperJ eq 2.
        # k_F = self._get_k_F_from_r(radius)
        k_max = self._get_k_max(radius)
        kapa_0 = math.sqrt(energy ** 2 + k_max ** 2 + self.IP)
        # eta = self._get_eta_from_r(radius, energy)
        # return 2 / np.pi * k_F * (0.5 + (1 - eta ** 2) / (4 * eta) * np.log(abs((1 + eta) / (1 - eta))))
        return 1 / (np.pi * kapa_0) * (0.5 * (kapa_0 ** 2 - k_max ** 2) * np.log((kapa_0 - k_max) / (kapa_0 + k_max)) + kapa_0 * k_max)
# modeling V_pol (polarisation term)
    def get_v_pol(self, radius):
        return (-self.alpha_0) * ((1 - np.exp(-1 * radius / self.rs)) ** 6 / (2 * radius ** 4))
# modeling V_abs (absorption term)
    def get_v_abs(self, radius, energy):
        if self.y_values is None and self.r1_values is None:
            self.update_r1_values()
        v_loc_r = energy ** 2 - \
            self.get_v_st(radius) - self.get_v_ex(radius, energy) - \
            self.get_v_pol(radius)
        # k_F = self._get_k_F_from_r(radius)
        k_F = self._get_k_max(radius)
        A1 = 5 * k_F ** 3 / (2 * self.delta)
        A2 = - k_F ** 3 * (5 * energy ** 2 - 3 * k_F ** 2) / \
            (energy ** 2 - k_F ** 2) ** 2
        temp = 2 * k_F ** 2 + 2 * self.delta - energy ** 2
        if temp <= 0:
            A3 = 0
        else:
            A3 = 2 * np.heaviside(2 * k_F ** 2 + 2 * self.delta - energy ** 2, 1) * \
                (pow(2 * k_F ** 2 + 2 * self.delta - energy ** 2, 2.5) / (energy ** 2 - k_F ** 2) ** 2)
        return -1 * self.get_rho_r(radius) * np.sqrt(v_loc_r / 2) * \
            (8 * np.pi / 5 * (energy ** 2) * (k_F ** 3)) * np.heaviside(2 * energy ** 2 - k_F ** 2 - self.delta * 2, 1) * (A1 + A2 + A3)
# getting Real Potential
    def get_v_R(self, radius, energy):
        if self.y_values is None and self.r1_values is None:
            self.update_r1_values()
        return self.get_v_st(radius) + self.get_v_ex(radius, energy) + self.get_v_pol(radius)
# We calculate the cross sections using varaible phase approach. Line 176-252 directly from Shouyi's code for calcualting phase shift.
    def get_phases(self, radius, energy, order, x_l):
        rb_first = riccati_bessel_first(order, radius * energy)
        rb_second = riccati_bessel_second(order, radius * energy)
        # PaperP eq 11a and 11b
        A = np.cosh(x_l[1]) * (rb_first * np.cos(x_l[0]) - rb_second * np.sin(x_l[0]))
        B = -np.sinh(x_l[1]) * (rb_first * np.sin(x_l[0]) - rb_second * np.cos(x_l[0]))
        # PaperP eq 9 and eq 10
        dreal_x_l_dkr = -(2.0/energy) * (2 * self.get_v_R(radius, energy) * (A ** 2 - B ** 2) + 2 * self.get_v_abs(radius, energy) * A * B)
        dimaginary_x_l_dkr = -(2.0/energy) * (2 * self.get_v_R(radius, energy) * A * B - 2 * self.get_v_abs(radius, energy) * (A ** 2 - B ** 2))

        return [dreal_x_l_dkr, dimaginary_x_l_dkr]

    def get_next_e(self, radius, energy, order, e_value, step):
        k_list = []
        r_list = [radius,
                radius + step / 2.0,
                radius + step / 2.0,
                radius + step]

        # First set
        new_phase_e = self.get_phases(r_list[0], energy, order, e_value)
        k_list.append([v * step for v in new_phase_e])   
        # Second set
        new_phase_e = self.get_phases(r_list[1], energy, order, e_value + [v / 2.0 for v in k_list[0]])
        k_list.append([v * step for v in new_phase_e])
        # Third set
        new_phase_e = self.get_phases(r_list[2], energy, order, e_value + [v / 2.0 for v in k_list[1]])
        k_list.append([v * step for v in new_phase_e])
        # Fourth set
        new_phase_e = self.get_phases(r_list[3], energy, order, e_value + k_list[2])
        k_list.append([v * step for v in new_phase_e])
        
        next_e = np.array(e_value) + np.array([v / 6.0 for v in k_list[0]]) + np.array([v / 3.0 for v in k_list[1]]) \
                + np.array([v / 3.0 for v in k_list[2]]) + np.array([v / 6.0 for v in k_list[3]])
        return next_e

    def integrate_function(self, energy, order):
        rmin = 0.01
        rmax = 50
        rstep = (rmax - rmin) / 5000.0
        e_values = []
        e_values.append([1.0, 0.0])

        for i in range(0, 5000):
            radius = rmin + i * rstep
            next_e = self.get_next_e(radius, energy, order, e_values[i], rstep)
            if np.isnan(next_e[0]) or np.isnan(next_e[1]):    
                break
            else:
                e_values.append(next_e)
        x_l = e_values[-1][0]
        x_l_bar = e_values[-1][1]
        ans = complex(x_l, x_l_bar)
        return np.exp(ans.imag) * np.exp(ans.real * 1j)

    def loop_over_partial_waves(self, energy):
        #unit conversion
        energy_hartree = energy * conversions.EV_TO_HARTREE
        energy_sq = energy_hartree * 2
        k_energy = np.sqrt(energy_sq) * np.sqrt(2)
        
        xs_total_i = 0.0
        xs_el_i = 0.0
        xs_abs_i = 0.0
        order_max =  3
        
        for order in range(0, order_max):
            phase_e = self.integrate_function(k_energy, order)
            l_xs_total_i = (2 * np.pi / k_energy ** 2) * (2.0 * order + 1.0) * (1 - phase_e.real)
            l_xs_el_i = (np.pi / k_energy ** 2) * (2 * order + 1) * abs(1 - phase_e) ** 2
            l_xs_abs_i = (np.pi / k_energy ** 2) * (2 * order + 1) * (1 - abs(phase_e)) ** 2
            # print(order, phase_e, l_xs_total_i, l_xs_el_i, l_xs_abs_i)
            xs_total_i += l_xs_total_i
            xs_el_i += l_xs_el_i
            xs_abs_i += l_xs_abs_i

        return xs_total_i  * conversions.BOHR_SQ_TO_ANGSTROM_SQ, xs_el_i * conversions.BOHR_SQ_TO_ANGSTROM_SQ, xs_abs_i * conversions.BOHR_SQ_TO_ANGSTROM_SQ 



    def coupled_differential_equations(self, kr, x_l):
        # PaperP 
        energy = self.energy
        radius = kr / energy
        rb_first = riccati_bessel_first(self.order, kr)
        rb_second = riccati_bessel_second(self.order, kr)
        # PaperP eq 11a and 11b
        A = np.cosh(x_l[1]) * (rb_first * np.cos(x_l[0]) - rb_second * np.sin(x_l[0]))
        B = -np.sinh(x_l[1]) * (rb_first * np.sin(x_l[0]) - rb_second * np.cos(x_l[0]))
        # PaperP eq 9 and eq 10
        dreal_x_l_dkr = -(2.0/energy) * (2 * self.get_v_R(radius, energy) * (A ** 2 - B ** 2) + 2 * self.get_v_abs(radius, energy) * A * B)
        dimaginary_x_l_dkr = -(2.0/energy) * (2 * self.get_v_R(radius, energy) * A * B - 2 * self.get_v_abs(radius, energy) * (A ** 2 - B ** 2))

        return [dreal_x_l_dkr, dimaginary_x_l_dkr]


    def get_S_l_k(self, energy, order, rmin=0.01, rmax=8.0, iternumber=1000):
        rstep = (rmax - rmin) / iternumber
        real_x_l = 0.0
        imaginary_x_l = 0.0
        
        self.order = order
        self.energy = energy
        
        y_vector_start = [real_x_l, imaginary_x_l]
        solver = integrate.RK45(self.coupled_differential_equations, rmin, y_vector_start, rmax, max_step=0.01, atol=1e-5, first_step=rstep)

        steps_max = iternumber
        for _ in range(steps_max):
            solver.step()
            if solver.status == 'finished':
                break

        self.order = energy = None
        return np.exp(-2 * solver.y[0]) * np.exp(complex(0, 2 *solver.y[1]))
  # calculate elatic cross section  
    def get_sigma_el(self, energy, l_min=0, l_max=30):
        sigma_el_sum = 0
        for l in range(l_min, l_max + 1):
            sigma_el_l = np.pi * (2 * l + 1) * abs(1 - self.get_S_l_k(energy, l)) ** 2 / energy ** 2
            sigma_el_sum = sigma_el_sum + sigma_el_l
        return sigma_el_sum
# calculate absorption cross sections
    def get_sigma_abs(self, energy, l_min=0, l_max=30):
        sigma_abs_sum = 0
        for l in range(l_min, l_max + 1):
            sigma_abs_l = np.pi * (2 * l + 1) * (1 - abs(self.get_S_l_k(energy, l))) ** 2 / energy ** 2
            sigma_abs_sum = sigma_abs_sum + sigma_abs_l
        return sigma_abs_sum
# Define some specific terms for different atoms and molecules
class Helium(Atom):
    def __init__(self, electron_num=2, delta=0.9, IP=0.9, alpha_0=0.64, rs=2.6456, charge_density_path='data/he_charge_density.dat'):
        
        # https://cccbdb.nist.gov/polcalc1x.asp
        super().__init__(electron_num, delta, IP, alpha_0, rs, charge_density_path)


class Neon(Atom):
    def __init__(self, electron_num=10, delta=0.79, IP=0.79, alpha_0=2.37, rs=2.91, charge_density_path='updated_charge_densities/ne_charge_density.dat'):
        
        super().__init__(electron_num, delta, IP, alpha_0, rs, charge_density_path)


class Water(Atom):
    def __init__(self, electron_num=10, delta=0.4, IP=0.44, alpha_0=8.35, rs=4.9, charge_density_path='updated_charge_densities/h2o_charge_density.dat'):
        super().__init__(electron_num, delta, IP, alpha_0, rs, charge_density_path)
    
    def get_v_st(self, radius, r1_values=None, y_values=None, epsilon=1e-6):
        if r1_values is None or y_values is None:
            r1_values = self.r1_values
            y_values = self.y_values
        y_list = []
        r1_list = []
        for i in range(len(r1_values)):
            r1 = r1_values[i]
            rho_r1 = y_values[i]

            if abs(radius - r1) < epsilon:
                continue
            else:
                r1_list.append(r1)
                y = rho_r1 / abs(radius - r1)
                y_list.append(y)
        part_1 = integrate.trapz(y_list, r1_list)
        part_2, err2 = integrate.dblquad(lambda y, x: (8 / np.sqrt((radius * np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x) + 0.0647) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y) - 0.74879) ** 2 + (radius * np.cos(x) - 0.514) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y) + 0.74879) ** 2 + (radius * np.cos(x) - 0.514) ** 2)) * np.sin(x),\
                        0, math.pi, 0, 2 * math.pi)


        return part_1 - part_2


class Methane(Atom):
    def __init__(self, electron_num=10, delta=0.386, IP=0.477, alpha_0=16.02, rs=6.3, charge_density_path='updated_charge_densities/ch4_charge_density.dat'):
        super().__init__(electron_num, delta, IP, alpha_0, rs, charge_density_path)

    def get_v_st(self, radius, r1_values=None, y_values=None, epsilon=1e-6):
        if r1_values is None or y_values is None:
            r1_values = self.r1_values
            y_values = self.y_values
        y_list = []
        r1_list = []
        for i in range(len(r1_values)):
            r1 = r1_values[i]
            rho_r1 = y_values[i]

            if abs(radius - r1) < epsilon:
                continue
            else:
                r1_list.append(r1)
                y = rho_r1 / abs(radius - r1)
                y_list.append(y)
        part_1 = integrate.trapz(y_list, r1_list)
        part_2, err2 = integrate.dblquad(lambda y, x: (8 / np.sqrt((radius * np.sin(x) * np.cos(y) - 0.8906223382) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x) - 0.6297655560) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y) + 0.8906223382) ** 2 + (radius * np.cos(x) + 0.6297655560) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y) + 0.8906223382) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x) - 0.6297655560) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y) - 0.8906223382) ** 2 + (radius * np.cos(x) + 0.6297655560) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x)) ** 2)) * np.sin(x),\
                        0, math.pi, 0, 2 * math.pi)


        return part_1 - part_2


class Nitrogen(Atom):
    def __init__(self, electron_num=14, delta=0.62, IP=0.57, alpha_0=11.12, rs=5.7, charge_density_path='updated_charge_densities/n2_charge_density.dat'):
        super().__init__(electron_num, delta, IP, alpha_0, rs, charge_density_path)

    def get_v_st(self, radius, r1_values=None, y_values=None, epsilon=1e-6):
        if r1_values is None or y_values is None:
            r1_values = self.r1_values
            y_values = self.y_values
        y_list = []
        r1_list = []
        for i in range(len(r1_values)):
            r1 = r1_values[i]
            rho_r1 = y_values[i]

            if abs(radius - r1) < epsilon:
                continue
            else:
                r1_list.append(r1)
                y = rho_r1 / abs(radius - r1)
                y_list.append(y)
        part_1 = integrate.trapz(y_list, r1_list)
        part_2, err2 = integrate.dblquad(lambda y, x: (8 / np.sqrt((radius * np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x) + 0.5386504532) ** 2) + \
                        1 / np.sqrt((radius *np.sin(x) * np.cos(y)) ** 2 + (radius * np.sin(x) * np.sin(y)) ** 2 + (radius * np.cos(x) - 0.5386504532) ** 2)) * np.sin(x),\
                        0, math.pi, 0, 2 * math.pi)


        return part_1 - part_2