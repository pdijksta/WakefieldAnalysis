import numpy as np
from scipy.constants import physical_constants, c
import matplotlib.pyplot as plt

import myplotstyle as ms


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



if __name__ == '__main__':

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

