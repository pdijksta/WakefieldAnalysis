import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import quad

import uwf_model
from uwf_model import IM, RE

import myplotstyle as ms

plt.close('all')

a = 2.2e-3
h = 100e-9
kappa = 2*pi/10e-6
deviation_im = 1e-2
deviation_until = 1.

r = uwf_model.r(h, kappa, a)


s_arr = np.array([5, 25, 50, 100, 150])*1e-6

#H = uwf_model.H(tau, r)

#w_surface_round_tube = uwf_model.surface_round_tube(s, a, kappa, h)


subplot = ms.subplot_factory(2,3)

for s in s_arr:
    ms.figure('s = %i um' % (s*1e6))
    sp_ctr = 1

    tau = np.array([kappa * s])
    for ctr, (lim1, lim2, args) in enumerate([
            (0., deviation_im, (IM, 0, 0, tau, r)),
            (0., deviation_until, (RE, 0, deviation_im, tau, r)),
            (deviation_im, 0, (IM, deviation_until, 0, tau, r)),
            (deviation_until, np.inf, (RE, 0, 0, tau, r)),
            ]):

        sp = subplot(sp_ctr, title=sp_ctr)
        sp_ctr += 1

        lim2 = min(lim2, 10)
        xx = np.linspace(lim1, lim2, 1000) # , dtype='complex128')
        yy = uwf_model.H_integrand(xx, *args)
        integral, error = quad(uwf_model.H_integrand, lim1, lim2, args, limit=100)
        print(ctr, s*1e6, integral, error, '%.4f' % (error/integral*100)+'%')

        sp.plot(xx, yy)

plt.show()

