import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

import data_loader
import uwf_model as uwf
import myplotstyle as ms

plt.close('all')

semigap = 4e-3/2.

data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
meas1 = 'Bunch_length_meas_2020-02-03_21-54-24.h5'
lims = [25, 70]
bl_meas = data_loader.load_blmeas(data_dir+meas1)
charge = 200e-12

charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
current_profile = bl_meas['current1']

charge_profile = current_profile * charge / np.sum(current_profile)
energy_eV = bl_meas['energy_eV']


mask_charge = np.logical_and(charge_xx > lims[0]*1e-6, charge_xx < lims[1]*1e-6)
assert np.any(mask_charge)
charge_xx = charge_xx[mask_charge]
charge_xx -= charge_xx.min()
charge_profile = charge_profile[mask_charge]

fig = ms.figure('Surface variable scan')
sp_ctr = 1
subplot = ms.subplot_factory(3,3)

sp_charge = subplot(sp_ctr, title='Current profile', xlabel='s [$\mu$m]')
sp_ctr += 1

sp_charge.plot(charge_xx*1e6, charge_profile)

sp_surface = subplot(sp_ctr, title='Surface WF', xlabel='s [$\mu$m]')
sp_ctr += 1


h_arr = [50e-9, 100e-9, 200e-9]
kappa_factor_arr = [0.5, 1, 2]

for h, kappa_factor in itertools.product(h_arr, kappa_factor_arr):
    kappa = uwf.aramis_kappa * kappa_factor
    result, err = uwf.surface_round_tube(charge_xx, semigap, kappa, h, multi=True)

    label = '%i %.1f' % (h*1e9, kappa_factor)
    sp_surface.plot(charge_xx*1e6, result, label=label)

sp_surface.legend()









plt.show()

