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

def wxd(s, a):
    """
    Single particle wake.
    Unit: V/m /m (offset)
    """
    t1 = Z0 * c / (4*pi)
    t2 = pi**4 / (16 * a**4)
    s0d_ = s0d(a)
    t3 = s0d_
    sqr = np.sqrt(s/s0d_)
    t4 = 1 - (1 + sqr)*np.exp(-sqr)
    return t1*t2*t3*t4

class WakeFieldCalculator:
    def __init__(self, xx, charge_profile):
        self.xx = xx
        self.charge_profile = charge_profile

    def get_single_particle_wake(self, semigap):
        return wxd(self.xx, semigap)

    def get_wake_potential(self, single_particle_wake):
        return np.convolve(self.charge_profile, single_particle_wake)[:len(self.xx)]

    def get_kick_factor(self, wake_effect):
        return np.sum(self.charge_profile*wake_effect) / np.sum(self.charge_profile)

    def get_kick(self, kick_factor, R12, beam_energy_eV):
        return (Ls * kick_factor * R12) / beam_energy_eV

    def calc_all(self, semigap, R12, beam_energy_eV):
        spw = self.get_single_particle_wake(semigap)
        wake_potential = self.get_wake_potential(spw)
        kick_factor = self.get_kick_factor(wake_potential)
        x_per_m_offset = self.get_kick(kick_factor, R12, beam_energy_eV)

        return {
                'single_particle_wake': spw,
                'wake_potential': wake_potential,
                'kick_factor': kick_factor,
                'x_per_m_offset': x_per_m_offset,
                'charge_xx': self.xx,
                'charge_profile': self.charge_profile
                }

#def convolve(xx, charge_profile, semigap):
#    single_particle_wake = wxd(xx, semigap)
#    wake_effect = np.convolve(charge_profile, single_particle_wake)[:len(xx)]
#    return single_particle_wake, wake_effect







#### OLD
#def get_matrix_drift(L):
#    return np.array([[1, L], [0, 1]], float)
#
#def get_matrix_quad(k1l, L):
#    sin, cos, sqrt = scipy.sin, scipy.cos, scipy.sqrt # numpy trigonometric functions do not work
#    k1 = k1l/L
#    phi = L * sqrt(k1)
#    mat = np.array([[cos(phi), sin(phi) / sqrt(k1)],
#    [-sqrt(k1) * sin(phi), cos(phi)]], dtype=complex)
#    return np.real(mat)
#
#def matmul(matrix_list):
#    output = matrix_list[0].copy()
#    for m in matrix_list[1:]:
#        output = output @ m
#    return output
#### END OLD



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import myplotstyle as ms

    plt.close('all')

    fig = ms.figure('Single particle wake')
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp = subplot(sp_ctr, title='Single particle wake', xlabel='s [m]', ylabel='V/nC/m', scix=True)
    sp_ctr += 1

    test_arr = np.linspace(0, 3e-3, int(1e3))

    for gap in np.arange(2, 10.01, 1)*1e-3:
        sp.semilogy(test_arr, wxd(test_arr, gap), label='%i mm gap' % (gap*1e3))

    sp.legend()

    plt.show()

