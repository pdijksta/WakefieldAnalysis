import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import myplotstyle as ms
from doublehornfit import DoublehornFit

plt.close('all')

with open('./test_profile_fit.pkl', 'rb') as f:
    profile = pickle.load(f)

def gauss(xx, scale, mean, sig, const):
    #return scale*stats.norm.pdf(xx, mean, sig)
    if sig != 0:
        return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const
    else:
        return 0


def double_horn(xx, pos_left, pos_right, s1a, s1b, s2a, s2b, const_middle, max_left, max_right):
    outp = np.zeros_like(xx)
    pos_middle = (pos_right + pos_left)/2.

    mask1a = xx < pos_left
    outp[mask1a] = gauss(xx[mask1a], max_left, pos_left, s1a, 0)

    mask1b = np.logical_and(xx > pos_left, xx < pos_middle)
    outp[mask1b] = gauss(xx[mask1b], max_left-const_middle, pos_left, s1b, const_middle)

    mask2a = np.logical_and(xx > pos_middle, xx < pos_right)
    outp[mask2a] = gauss(xx[mask2a], max_right-const_middle, pos_right, s2a, const_middle)

    mask2b = xx > pos_right
    outp[mask2b] = gauss(xx[mask2b], max_right, pos_right, s2b, 0)

    return outp

ms.figure('Test fit')
subplot = ms.subplot_factory(1, 1)

sp = subplot(1)

sp.plot(profile._xx, profile._yy)

pos_left = -40e-15
pos_right = 48e-15
s1a = s1b = s2a = s2b = 8e-15

const_middle = 8.9e-14
max_left = 1.4e-13
max_right = 2.1e-13

xx_fit = profile.time
p0 = [pos_left, pos_right, s1a, s1b, s2a, s2b, const_middle, max_left, max_right]
yy_p0 = double_horn(xx_fit, *p0)

popt, pcov = curve_fit(double_horn, profile.time, profile.current, p0)
yy_fit = double_horn(xx_fit, *popt)


gf = profile.gaussfit

arg_left = np.argmax(gf.yy[gf.xx < gf.mean])
pos_left = gf.xx[arg_left]
max_left = gf.yy[arg_left]


mask_right = gf.xx > gf.mean
arg_right = np.argmax(gf.yy[mask_right]) + np.sum(mask_right == 0)
pos_right = gf.xx[arg_right]
max_right = gf.yy[arg_right]
s1a = s1b = s2a = s2b = gf.sigma / 5

const_middle = np.min(gf.yy[np.logical_and(gf.xx > pos_left, gf.xx < pos_right)])

p0_auto = [pos_left, pos_right, s1a, s1b, s2a, s2b, const_middle, max_left, max_right]
yy_p0_auto = double_horn(xx_fit, *p0_auto)

dhf = DoublehornFit(profile.time, profile.current)


#sp.plot(xx_fit, yy_p0, label='Guess')
sp.plot(dhf.xx, dhf.fit_func(dhf.xx, *dhf.p0), label='Guess')
#sp.plot(xx_fit, yy_fit, label='Fit')
sp.plot(dhf.xx, dhf.reconstruction, label='Fit')

sp.legend()

plt.show()

