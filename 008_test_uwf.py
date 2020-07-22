import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import uwf_model
from EmittanceTool.h5_storage import loadH5Recursive
import myplotstyle as ms


h = 100e-9
kappa = 2*np.pi/10e-6

plt.close('all')

#array_size = int(1e2)
#charge_profile = np.ones([array_size])
#charge_profile *= 200e-12 / (np.sum(charge_profile))
#s_arr = np.linspace(0, 100e-6, array_size)

example = loadH5Recursive('./example_current_profile.h5')
s_arr = example['time_profile']*c
s_arr -= s_arr.min()
charge_profile = example['current']
charge_profile = charge_profile / charge_profile.sum() * 200e-12

gap_list = [4e-3, 4.4e-3, 8e-3, 10e-3]
wf_list = []
wf_err_list2 = []
wf_list2 = []
conv_list = []
conv_list2 = []
for gap in gap_list:
    #ac_round_tube = uwf_model.ac_round_tube(s_arr, gap)
    #surface_round_tube, err = uwf_model.surface_round_tube(s_arr, gap/2, kappa, h)

    uwf_dict = uwf_model.calc_all(s_arr, charge_profile, gap/2)
    ac_round_tube = uwf_dict['w_ac']
    surface_round_tube = uwf_dict['w_surface']
    surface_err = uwf_dict['w_surface_err']
    W_ac = uwf_dict['W_ac']
    W_surface = uwf_dict['W_surface']

    wf_list.append(ac_round_tube)
    wf_list2.append(surface_round_tube)
    wf_err_list2.append(surface_err)
    conv_list.append(W_ac)
    conv_list2.append(W_surface)
    #break

ms.figure('Undulator wakefield model')
subplot = ms.subplot_factory(2, 3)
sp_ctr = 1

xlabel = 's [$\mu$m]'
ylabel = 'w(kV/(pC$\cdot$m))'
sp_charge = subplot(sp_ctr, title='Charge profile', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1
sp_charge.plot(s_arr*1e6, charge_profile)

sp_wf = subplot(sp_ctr, title='Resistive wake', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1
sp_wf2 = subplot(sp_ctr, title='Surface roughness wake', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1


ylabel2 = 'W(kV/m)'
sp_wf_W = subplot(sp_ctr, title='Resisistive Wake convolved', xlabel=xlabel, ylabel=ylabel2)
sp_ctr += 1


sp_wf2_W = subplot(sp_ctr, title='Surface Roughness convolved', xlabel=xlabel, ylabel=ylabel2)
sp_ctr += 1

conversion_factor = 1e-3*1e-12
for gap, wf, wf2, err2, conv1, conv2 in zip(gap_list, wf_list, wf_list2, wf_err_list2, conv_list, conv_list2):
    label = '%.1f' % (gap*1e3)
    sp_wf.plot(s_arr*1e6, wf*conversion_factor, label=label)
    sp_wf2.errorbar(s_arr*1e6, wf2*conversion_factor, label=label, yerr=err2*conversion_factor)
    #conv1 = uwf_model.convolve(charge_profile, wf)
    sp_wf_W.plot(s_arr*1e6, conv1*1e-3, label=label)

    #conv2 = uwf_model.convolve(charge_profile, wf2)
    sp_wf2_W.plot(s_arr*1e6, conv2*1e-3, label=label)

sp_wf.legend(title='Undulator gap')
sp_wf2.legend(title='Undulator gap')
sp_wf_W.legend(title='Undulator gap')


#sp_integrand = subplot(sp_ctr,
#sp_ctr += 1



plt.show()



