import functools
import numpy as np
#import scipy
import scipy.integrate as integrate
from scipy.constants import c, physical_constants
from scipy import sqrt
from numpy import pi, exp

Z0 = physical_constants['characteristic impedance of vacuum'][0]
conversion_factor = Z0 * c / (4*pi)
IM, RE, BOTH = 0, 1, 2
quad = integrate.quad



# Bane2004

# radius a
# Conductivity sigma

#sigma_cu = 5.96e7
sigma_cu = 5.8e17 # cgs units arrrgh
ctau_cu = 8.1e-6 # from "Resistive wall wakefield in the LCLS undulator beam pipe", Bane and Stupakov, SLAC-PUB-10707, 2004

def s0(a, sigma):
    return ((c*a**2) / (2*pi*sigma))**(1/3)

def dc_round_tube(s, gap, sigma=sigma_cu):
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

def impedance_flat_integrand(q, a, lamb, k, im_or_re):
    if k == 0:
        return 0.
    arg = q*a
    cosh = np.cosh(arg)
    t1 = lamb/k * cosh**2
    if q == 0:
        t2 = 1j*k*a*cosh # sinh(q*a)/q = a for q = 0
    else:
        t2 = 1j*k/q * np.sinh(arg) * cosh
    outp = 1./(t1 - t2)

    if im_or_re == RE:
        return np.real(outp)
    elif im_or_re == IM:
        return np.imag(outp)
    elif im_or_re == BOTH:
        return outp

def impedance_flat_ac(k, a, sigma=sigma_cu, ctau=ctau_cu, lim=(-10e3, 10e3)):
    sigma_tilde = sigma / (1 - 1j*k*ctau)
    lamb = sqrt((2*pi*sigma_tilde*np.abs(k))/c) * (1j + np.sign(k))

    integral_re, err_re = quad(impedance_flat_integrand, lim[0], lim[1], (a, lamb, k, RE))
    integral_im, err_im = quad(impedance_flat_integrand, lim[0], lim[1], (a, lamb, k, IM))

    pre_factor = 1/c

    return integral_re * pre_factor, integral_im * pre_factor, err_re * pre_factor, err_im * pre_factor

def wake_flat_ac(s_arr, a, sigma=sigma_cu, ctau=ctau_cu, lim_imp=(-10e3, 10e3), k_factor=10, k_size=2**13):
    s0_ = s0(a, sigma)
    k_arr = np.linspace(0, k_factor/s0_, k_size)
    re_impedance = np.zeros_like(k_arr)
    for n_k, k in enumerate(k_arr):
        sigma_tilde = sigma / (1 - 1j*k*ctau)
        lamb = sqrt((2*pi*sigma_tilde*np.abs(k))/c) * (1j + np.sign(k))

        integral_re, err_re = quad(impedance_flat_integrand, lim_imp[0], lim_imp[1], (a, lamb, k, RE))
        re_impedance[n_k] = integral_re

    re_impedance /= c

    W_arr = np.zeros_like(s_arr)
    for n_s, s in enumerate(s_arr):
        cos_integrand = re_impedance * np.cos(k_arr*s)
        W_arr[n_s] = np.trapz(cos_integrand, k_arr)
    W_arr *= 2*c/np.pi

    result_dict = {
            'W_arr': W_arr,
            'k_arr': k_arr,
            're_impedance': re_impedance,
            's_arr': s_arr,
            }
    return result_dict

def impedance_flat_re_dc(k, a, sigma=sigma_cu, lim=(-10e3, 10e3)):
    lamb = sqrt((2*pi*sigma*np.abs(k))/c) * (1j + np.sign(k))
    integral_re, err_re = quad(impedance_flat_integrand, lim[0], lim[1], (a, lamb, k, RE))
    integral_im, err_im = quad(impedance_flat_integrand, lim[0], lim[1], (a, lamb, k, IM))

    return integral_re/c, integral_im/c

def integrand_wake_flat_cos_dc(k, a, s, sigma=sigma_cu):
    z = impedance_flat_re_dc(k, a, sigma)
    return z * np.cos(k*s)

def integrand_wake_flat_fourier(k, a, s, sigma=sigma_cu):
    z = impedance_flat_re_dc(k, a, sigma)
    return z * np.exp(1j*k*s)

def wake_flat_cos_dc(s, semigap, sigma=sigma_cu):
    factor = 2*c / pi
    integral, err = quad(integrand_wake_flat_cos_dc, 0, np.inf, (semigap, s, sigma))
    return factor * integral, factor*err

def wake_flat_fourier(s, semigap, sigma=sigma_cu):
    factor = 2*c / pi
    integral, err = quad(integrand_wake_flat_fourier, -np.inf, np.inf, (semigap, s, sigma))
    return factor * integral, factor*err

# Aramis undulator constants
aramis_h = 100e-9
aramis_kappa = 2*np.pi/10e-6

def calc_all(s, charge_profile, semigap, kappa=aramis_kappa, h=aramis_h, L=1.):
    w_surface, w_surface_err = surface_round_tube(s, semigap, kappa, h)
    #w_ac = dc_round_tube(s, semigap*2)
    w_ac = wake_flat_ac(s, semigap)

    W_surface = convolve(charge_profile, w_surface) * L
    W_ac = convolve(charge_profile, w_ac) * L

    proj_Eloss_surface = np.sum(charge_profile * W_surface) / np.sum(charge_profile)
    proj_Eloss_dc = np.sum(charge_profile * W_ac) / np.sum(charge_profile)

    proj_Espread_surface = calc_espread(W_surface, charge_profile)
    proj_Espread_dc = calc_espread(W_ac, charge_profile)

    outp = {
            'w_surface': w_surface,
            'w_surface_err': w_surface_err,
            'W_surface': W_surface,
            'w_ac': w_ac,
            'W_ac': W_ac,
            'proj_Eloss_surface': proj_Eloss_surface,
            'proj_Eloss_dc': proj_Eloss_dc,
            'proj_Espread_surface': proj_Espread_surface,
            'proj_Espread_dc': proj_Espread_dc,
            's': s,
            'charge_profile': charge_profile,
            'kappa': kappa,
            'h': h,
            'L': L,
            }
    return outp

