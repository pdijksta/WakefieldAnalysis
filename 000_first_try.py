import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from wf_model import wxd, Ls

from h5_storage import loadH5Recursive, saveH5Recursive
import myplotstyle as ms

plt.close('all')


home_office = True
load_compact = True
storage_dir = '/storage/'

if home_office:
    def loadH5Recursive2(path):
        return loadH5Recursive(os.path.join(storage_dir + 'data_2020-02-03/', os.path.basename(path)))
else:
    loadH5Recursive2 = loadH5Recursive

bl_meas_structure = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-51-34.h5'


if load_compact:
    bl_meas_compact = loadH5Recursive('./example_current_profile.h5')
    energy_eV = bl_meas_compact['energy_eV']
    time_profile = bl_meas_compact['time_profile']
    current = bl_meas_compact['current']
else:
    bl_meas = loadH5Recursive2(bl_meas_structure)
    time_profile, current = bl_meas['Meta_data']['Time axes'][::-1]*1e-15, bl_meas['Meta_data']['Current profile'][::-1]
    energy_eV = bl_meas['Input']['Energy'].squeeze()*1e6
    saveH5Recursive('./example_current_profile.h5', {'time_profile': time_profile, 'current': current, 'energy_eV': energy_eV})


rescale_factor = 200e-12/np.sum(current)

yy_charge = current * rescale_factor
xx_space = time_profile*c
xx_space -= xx_space.min()

#cutoff
#yy_charge[yy_charge < 0.1*yy_charge.max()] = 0


fig = ms.figure('Wake effect')
subplot = ms.subplot_factory(2,3)
sp_ctr = 1

sp_charge = subplot(sp_ctr, title='Charge', xlabel='s [m]', ylabel='Charge [C]', scix=True, sciy=True)
sp_ctr += 1

sp_charge.plot(xx_space, yy_charge)

sp_single = subplot(sp_ctr, title='Single particle wake', xlabel='s [$\mu$m]', ylabel='Wake [MV/(m nC) / m](offset)', scix=True)
sp_ctr += 1

sp_wake_effect = subplot(sp_ctr, title='Convoluted wake', xlabel='s [$\mu$m]', ylabel='E [MV/m m(offset)]', scix=True)
sp_ctr += 1

sp_wake_effect2 = subplot(sp_ctr, title='Convoluted wake', xlabel='s [$\mu$m]', ylabel='E [MV/m m(offset)]', scix=True)
sp_ctr += 1

sp_kick_factor = subplot(sp_ctr, title='Kick factor', xlabel='Gap [mm]', ylabel='Kick factor [MV/m m(offset)]', scix=True, sciy=False)
sp_ctr  += 1

sp_delta_x  = subplot(sp_ctr, title='$\Delta$ x assuming $R_{12}$=10 m', xlabel='Offset [mm]', ylabel='$\Delta$ x [$\mu$m]', scix=True, sciy=False)
sp_ctr += 1

offset = np.array([-1e-3, 0, 1e-3])

delta_x_list = []
kick_list = []
gap_list = np.array([2.5e-3, 4e-3, 6e-3])

for gap in gap_list:
    label = gap*1e3
    single_particle_wake = wxd(xx_space, gap/2)
    sp_single.plot(xx_space*1e6, single_particle_wake*1e-15,label=label)
    wake_effect = np.convolve(yy_charge, single_particle_wake)[:len(xx_space)]

    #wake_effect2 = np.zeros_like(xx_space)
    #for n in range(len(xx_space)):
    #    for n2 in range(0,n):
    #        wake_effect2[n] += yy_charge[n2] * single_particle_wake[n-n2]
    #assert np.all(wake_effect == wake_effect2)

    sp_wake_effect.semilogy(xx_space*1e6, wake_effect*1e-6, label=label)
    sp_wake_effect2.plot(xx_space*1e6, wake_effect*1e-6, label=label)
    q = np.sum(yy_charge)
    kick_factor = np.sum(wake_effect*yy_charge) / q
    kick_list.append(kick_factor)
    delta_x = kick_factor * Ls * 10 / energy_eV
    delta_x_list.append(delta_x)
    print(gap, delta_x)
    sp_delta_x.plot(offset*1e3, delta_x*offset*1e6, marker='.', label=label)

sp_kick_factor.plot(gap_list*1e3, np.array(kick_list)*1e-6, marker='.')

for sp_ in sp_wake_effect, sp_single, sp_delta_x:
    sp_.legend(title='Gap size [mm]')

ms.saveall(storage_dir + '/plots/wakefield', wspace=0.3)

plt.show()

