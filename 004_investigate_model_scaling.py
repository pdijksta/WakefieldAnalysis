import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

import wf_model
import myplotstyle as ms

plt.close('all')

size = 1e4
total_charge = 200e-12
r12 = 11.5
energy_eV = 4e9
beam_offset = 1e-3

xx_space = np.linspace(0, 60e-6, int(size))
yy_charge = np.ones_like(xx_space)*total_charge/(size-2)
yy_charge[0] = 0
yy_charge[-1] = 0

wf_obj = wf_model.WakeFieldCalculator(xx_space, yy_charge)

fig = ms.figure('Ideal beam wake field')
subplot = ms.subplot_factory(2,2)

xlabel = 's [$\mu$m]'

sp_charge = subplot(1, title='Charge_profile (total = %i pC)' % (total_charge*1e12), xlabel=xlabel, ylabel='Charge (pC)', sciy=True)
sp_charge.plot(xx_space*1e6, yy_charge*1e12)

sp_wake = subplot(2, title='Single particle wake', xlabel=xlabel, ylabel='Single particle wake [MV/m m / nC]', sciy=True)

sp_delta_x = subplot(3, title='Offset after R12=10m and beam offset=1 mm', xlabel='Gap size [mm]', ylabel='Offset [mm]')


x_list = []
gap_list = np.exp(np.linspace(np.log(2.5e-6), np.log(10e-3), 20))
for gap_m in gap_list:
    semigap_m = gap_m/2

    wf_dict = wf_obj.calc_all(semigap_m, r12, energy_eV)
    delta_x = wf_dict['x_per_m_offset'] * beam_offset

    x_list.append(delta_x)
    wake = wf_dict['single_particle_wake']
    sp_wake.plot(xx_space*1e6, wake*1e-15, label='%.1e' % (gap_m*1e3))


x_list = np.array(x_list)

sp_delta_x.loglog(gap_list*1e3, x_list*1e3, marker='.', label='Data')

def order_func(xx, const, order):
    return const/xx**order

def quad_func(xx, const):
    return order_func(xx, const, 2)

for (label, func) in [('Quadratic', quad_func), ('Scaling', order_func)]:
    fit = opt.curve_fit(func, gap_list, x_list)[0]

    gap_plot = np.linspace(gap_list.min(), gap_list.max(), 100)

    if func == order_func:
        label += ' n= %.1f' % fit[1]
    sp_delta_x.loglog(gap_plot*1e3, func(gap_plot, *fit)*1e3, label=label)


sp_wake.legend(title='Gap [mm]')
sp_delta_x.legend()

ms.saveall('~/Dropbox/plots/004_model', ending='.pdf')

plt.show()

