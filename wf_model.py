#import scipy
import numpy as np
from scipy.constants import physical_constants, c



Z0 = physical_constants['characteristic impedance of vacuum'][0]
pi = np.pi

## Dimensions from Paolo

# a         variable semi-gap
# delta     corrugation depth
# p         period
# g         longitudinal gap
# w         plate width
# Ls        Length of streaker

Ls = 1
delta = 250e-6
p = 500e-6
g = 250e-6
w = 10e-3


## Functions from Paolo

alpha = 1 - 0.465*np.sqrt(g/p) - 0.070*g/p

def s0r(a):
    return (a**2 * delta) / (2*pi * alpha**2 * p**2)

def s0d(a):
    return s0r(a) * (15/14)**2

def s0yd(a, x):
    arg = pi*x/a
    return 4*s0r(a) * (3/2 + arg * 1/np.sin(arg) - arg/(2*np.tan(arg)))**(-2)

t1 = Z0 * c / (4*pi)
def wxd_lin_dipole(s, a):
    """
    Single particle wake, linear dipole approximation.
    Unit: V/m /m (offset)
    """
    t2 = pi**4 / (16 * a**4)
    s0d_ = s0d(a)
    t3 = s0d_
    sqr = np.sqrt(s/s0d_)
    t4 = 1 - (1 + sqr)*np.exp(-sqr)
    return t1*t2*t3*t4

def wxd_dipole(s, a, x):
    t2 = pi**3 / (4*a**3)
    arg = pi*x/(2*a)
    t3 = 1./np.cos(arg)**2
    t4 = np.tan(arg)
    t5 = s0yd(a, x)
    s0d_ = s0d(a)
    sqr = np.sqrt(s/s0d_)
    t6 = 1 - (1 + sqr)*np.exp(-sqr)
    return t1 * t2 * t3 * t4 * t5 * t6


class WakeFieldCalculator:
    def __init__(self, xx, charge_profile, beam_energy_eV):
        self.xx = xx
        self.charge_profile = charge_profile
        self.beam_energy_eV = beam_energy_eV

    def wake_potential(self, single_particle_wake):
        return np.convolve(self.charge_profile, single_particle_wake)[:len(self.xx)]

    def kick_factor(self, wake_potential):
        return np.sum(self.charge_profile*wake_potential) / np.sum(self.charge_profile)

    def kick(self, kick_factor, R12):
        return (Ls * kick_factor * R12) / self.beam_energy_eV

    def calc_all(self, semigap, R12, beam_offset=None, calc_lin_dipole=True, calc_dipole=True):
        output = {'input': {
                'semigap': semigap,
                'r12': R12,
                'beam_offset': beam_offset,
                'beam_energy_eV': self.beam_energy_eV,
                'charge_xx': self.xx,
                'charge_profile': self.charge_profile,
                }}

        if calc_lin_dipole:
            spw = wxd_lin_dipole(self.xx, semigap)
            wake_potential = self.wake_potential(spw)
            kick_factor = self.kick_factor(wake_potential)
            x_per_m_offset = self.kick(kick_factor, R12)

            output['lin_dipole'] = {
                    'single_particle_wake': spw,
                    'wake_potential': wake_potential,
                    'kick_factor': kick_factor,
                    'x_per_m_offset': x_per_m_offset,
                    }

        if calc_dipole:
            assert beam_offset is not None
            spw = wxd_dipole(self.xx, semigap, beam_offset)
            wake_potential = self.wake_potential(spw)
            kick_factor = self.kick_factor(wake_potential)
            x_per_m_offset = self.kick(kick_factor, R12)

            output['dipole'] = {
                    'single_particle_wake': spw,
                    'wake_potential': wake_potential,
                    'kick_factor': kick_factor,
                    'x_per_m_offset': x_per_m_offset,
                    }

        return output

