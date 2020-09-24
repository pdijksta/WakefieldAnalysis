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
            (deviation_until, 20, (RE, 0, 0, tau, r)),
            ]):

        sp = subplot(sp_ctr, title=sp_ctr)
        sp_ctr += 1

        lim2 = min(lim2, 20)
        xx = np.linspace(lim1, lim2, 1000) # , dtype='complex128')
        yy = uwf_model.H_integrand(xx, *args)
        integral, error = quad(uwf_model.H_integrand, lim1, lim2, args, limit=100)
        if ctr == 3:
            for n_steps in (1e3, 1e4, 1e5):
                xx2 = np.linspace(lim1, lim2, int(n_steps))
                #yy2 = np.array([uwf_model.H_integrand(x, *args) for x in xx2])
                yy2 = uwf_model.H_integrand(xx2, *args)
                int2 = np.trapz(yy2, xx2)
                xx3 = np.linspace(lim2, 2*lim2, int(n_steps))
                yy3 = uwf_model.H_integrand(xx3, *args)
                int3 = np.trapz(yy3, xx3)
                xx4 = np.linspace(2*lim2, 4*lim2, int(n_steps))
                yy4 = uwf_model.H_integrand(xx4, *args)
                int4 = np.trapz(yy4, xx4)
                print(ctr, s*1e6, int(np.log10(n_steps)), '%.1e' % int2, '%.1e' % int3, '%.1e' % int4, '%.4f' % (error/integral*100)+'%')
            sp2 = subplot(sp_ctr, title='%i B' % sp_ctr)
            sp2.plot(xx3, yy3)

            sp3 = subplot(sp_ctr, title='%i C' % sp_ctr)
            sp3.plot(xx4, yy4)

        sp.plot(xx, yy)

plt.show()

