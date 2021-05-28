import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import wf_model

import myplotstyle as ms

plt.close('all')

gap = 10e-3
distance_arr = np.linspace(200e-6, 500e-6, int(1e4))

ms.figure('Scaling')
sp = ms.subplot_factory(1,1)(1, xlabel='Distance [um]')

for s in [1e-6, 10e-6, 25e-6, 50e-6, 75e-6]:
    wake = wf_model.wxd(s, gap/2., gap/2. - distance_arr)
    color = sp.plot(distance_arr*1e6, wake, label='s=%i um' % (s*1e6))[0].get_color()

    def fit_func(x, scale, order):
        return scale * 1/x**order

    def fit_func3(x, scale):
        return scale * 1/x**3

    fit, _ = curve_fit(fit_func, distance_arr, wake, p0=[1e12, 1])
    fit2, _ = curve_fit(fit_func3, distance_arr, wake, p0=[1e12])
    reconstruction = fit_func(distance_arr, *fit)
    reconstruction2 = fit_func3(distance_arr, *fit2)
    sp.plot(distance_arr*1e6, reconstruction, label='Fit %.2f' % fit[1], ls='--', color=color)
    sp.plot(distance_arr*1e6, reconstruction2, label='Fit 3', ls='dotted', color=color)

sp.legend()


plt.show()

