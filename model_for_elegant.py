#import scipy
#import numpy as np
from scipy.constants import physical_constants, c
from numpy import cos, sin, tan, exp, sqrt, pi

Z0 = physical_constants['characteristic impedance of vacuum'][0]

## Dimensions from Paolo

# a         variable semi-gap
# delta     corrugation depth
# p         period
# g         longitudinal gap
# w         plate width
# Ls        Length of streaker

#Ls = 1.
delta = 250e-6
p = 500e-6
g = 250e-6
w = 10e-3


## Functions from Paolo

alpha = 1. - 0.465*sqrt(g/p) - 0.070*g/p

def s0r(a):
    return (a**2 * g) / (2*pi * alpha**2 * p**2)

t1 = Z0 * c / (4*pi)
def s0yd(a, x):
    arg = pi*x/a
    return 4*s0r(a) * (3/2 + arg * 1/sin(arg) - arg/(2*tan(arg)))**(-2)

def wxd(s, a, x, Ls=1.):
    t2 = pi**3 / (4*a**3)
    arg = pi*x/(2*a)
    t3 = 1./cos(arg)**2
    t4 = tan(arg)
    t5 = s0yd(a, x)
    sqr = sqrt(s/t5)
    t6 = 1 - (1 + sqr)*exp(-sqr)
    return t1 * t2 * t3 * t4 * t5 * t6 * Ls

def s0l(a, x):
    arg = (pi*x)/(2*a)
    return 4*s0r(a) * (1. + 1./3.*cos(arg)**2 + arg*tan(arg))**(-2)

def wld(s, a, x, Ls=1.):
    t2 = pi**2/(4*a**2)
    t3 = cos((pi*x)/(2*a))**(-2)
    s0l_ = s0l(a, x)
    t4 = exp(-sqrt(s/s0l_))
    return t1 * t2 * t3 * t4 * Ls

