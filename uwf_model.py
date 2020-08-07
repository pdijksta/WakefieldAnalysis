import functools
import numpy as np
#import scipy
import scipy.integrate as integrate
from scipy.constants import c, physical_constants
from scipy import sqrt
from numpy import pi, exp

Z0 = physical_constants['characteristic impedance of vacuum'][0]
conversion_factor = Z0 * c / (4*pi)
IM, RE = 0, 1
quad = integrate.quad



# Bane2004

# radius a
# Conductivity sigma

#sigma_cu = 5.96e7
sigma_cu = 5.8e17 # cgs units arrrgh

def s0(a, sigma):
    return ((c*a**2) / (2*pi*sigma))**(1/3)

def ac_round_tube(s, gap, sigma=sigma_cu):
    a = gap/2
    s0_ = s0(a, sigma)

    def integrand(x, s):
        return (x**2 * np.exp(-x**2 * s/s0_))/(x**6+8)

    t1 = 16/a**2
    t2 = exp(-s/s0_)/3 * np.cos(sqrt(3)*s/s0_)
    t3 = -sqrt(2)/pi
    t4 = np.zeros_like(s)
    for n_s, ss in enumerate(s):
        result, err = quad(integrand, 0, np.inf, args=ss)
        t4[n_s] = result

    #import pdb; pdb.set_trace()
    # negative sign for energy loss at s=0
    return -conversion_factor * t1*(t2 - t3 * t4)

def S(q):
    return q * (sqrt(2*q+1) - 1j*sqrt(2*q-1)) / sqrt(4*q**2-1)

def r(h, kappa, a):
    return 1./8. * h**2 * kappa**3 * a

#def complex_quadrature(func, a, b, *args, **kwargs):
#    def real_func(*args):
#        return scipy.real(func(*args))
#    def imag_func(*args):
#        return scipy.imag(func(*args))
#    real_integral = quad(real_func, a, b, *args, **kwargs)
#    imag_integral = quad(imag_func, a, b, *args, **kwargs)
#    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])


def H_integrand(var, im_or_re, const_re, const_im, tau, r):
    if im_or_re == IM:
        q = 1j * var
    else:
        q = var
    q = q + const_re + 1j*const_im
    t1 = exp(-1j*q*tau)
    sq = S(q)
    t2 = sq + 1
    t3 = 1 - 1j*r*q*sq
    t4 = 1 + 1j*r*q
    return -(t1 * t2/(t3*t4)).real


@functools.lru_cache(10000)
def H(tau, r, deviation_im=1e-2, deviation_re=1.):

    #deviation_im = 0.0001
    #deviation_re = 1
    outp1 = quad(H_integrand, 0., deviation_im, (IM, 0, 0, tau, r))
    outp2 = quad(H_integrand, 0., deviation_re, (RE, 0, deviation_im, tau, r))
    outp3 = quad(H_integrand, deviation_im, 0, (IM, deviation_re, 0, tau, r))
    outp4 = quad(H_integrand, deviation_re, np.inf, (RE, 0, 0, tau, r))

    int1 = outp1[0]
    int2 = outp2[0]
    int3 = outp3[0]
    int4 = outp4[0]
    int_total = int1 + int2 + int3 + int4

    err1 = outp1[1]
    err2 = outp2[1]
    err3 = outp3[1]
    err4 = outp4[1]
    err_total = err1 + err2 + err3 + err4

    factor = r/pi*conversion_factor
    return int_total * factor, err_total * factor

def surface_round_tube(s, semigap, kappa, h):
    """
    Considering sinusoidal corrugation

    s: distance between source and test particle (s>0)
    a: semigap
    kappa: wavenumber of corrugation (2pi/kappa is period)
    h: (depth of corrugation)
    """
    assert np.all(s >= 0)
    r_ = r(h, kappa, semigap)
    tau = s*kappa

    outp = np.zeros_like(s)
    err = np.zeros_like(s)
    for n_t, t in enumerate(tau):
        outp[n_t], err[n_t] = H(t, r_)
    factor = 4/semigap**2
    return factor * outp, factor * err

def convolve(charge_profile, single_particle_wake):
    return np.convolve(charge_profile, single_particle_wake)[:len(charge_profile)]

def calc_espread(wake_potential, charge_profile):
    total_charge = np.sum(charge_profile)
    full_variance = np.sum(wake_potential**2 * charge_profile) / total_charge
    eloss = np.sum(wake_potential * charge_profile) / total_charge
    eloss_sq = eloss**2
    result = np.sqrt(full_variance - eloss_sq)
    return result

# Aramis undulator constants
aramis_h = 100e-9
aramis_kappa = 2*np.pi/10e-6

def calc_all(s, charge_profile, semigap, kappa=aramis_kappa, h=aramis_h, L=1.):
    w_surface, w_surface_err = surface_round_tube(s, semigap, kappa, h)
    w_ac = ac_round_tube(s, semigap*2)

    W_surface = convolve(charge_profile, w_surface) * L
    W_ac = convolve(charge_profile, w_ac) * L

    proj_Eloss_surface = np.sum(charge_profile * W_surface) / np.sum(charge_profile)
    proj_Eloss_ac = np.sum(charge_profile * W_ac) / np.sum(charge_profile)

    proj_Espread_surface = calc_espread(W_surface, charge_profile)
    proj_Espread_ac = calc_espread(W_ac, charge_profile)

    outp = {
            'w_surface': w_surface,
            'w_surface_err': w_surface_err,
            'W_surface': W_surface,
            'w_ac': w_ac,
            'W_ac': W_ac,
            'proj_Eloss_surface': proj_Eloss_surface,
            'proj_Eloss_ac': proj_Eloss_ac,
            'proj_Espread_surface': proj_Espread_surface,
            'proj_Espread_ac': proj_Espread_ac,
            's': s,
            'charge_profile': charge_profile,
            'kappa': kappa,
            'h': h,
            'L': L,
            }
    return outp

