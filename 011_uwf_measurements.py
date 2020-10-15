import numpy as np
from scipy.io import loadmat
from scipy.constants import c
import matplotlib.pyplot as plt

import data_loader
#import uwf_model as uwf
import myplotstyle as ms

plt.close('all')

# From Elog entry https://elog-gfa.psi.ch/SwissFEL+commissioning/13404

# Beam energy was: 4042 MeV (SARBD01-MBND100:P-SET)

"""
beam setup 1 (about 40 fs)

    Eloss_UNDbis.mat (streak with positive offset of the structure) [21:35]
    Eloss_UND2ndStreak.mat (streak with negative offset) [21:47]

beam setup 2 (about 70 fs)

    Eloss_UND-COMP2.mat [22:24]

The images need post processing, since the gaussfits arent that great much of the time.

Same measurement for the gap of the first structure, still with compression setting 2

Eloss_DEH1-COMP2.mat [22:41]

Repeat with compression setting 1

Eloss_DEH1-COMP1.mat [22:53]

At compression setting 3: nothing saved because we believe we didnt transport the charge to the dump

Real compression setting 3 (100 pC):

Eloss_UND-COMP3.mat [00:19 (Feb4)]

Eloss_DEH1-COMP3.mat [00:30 (Feb4)

The three machine working points for the compression:

    S10 section phase: 69.68
    S10 section phase: 70.75??
    S10 section phase: 65.74 (overcompression for Gaussian current profile)
    (real 3) 100 pC ; S10 section phase 72.68
"""


# Undulator length: 12 modules, 4 m per module = 48 m

energy = 4042e6
data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
#data_dir = '/storage/data_2020-02-03/'

mat1 = data_dir + 'Eloss_UNDbis.mat'
dd = loadmat(mat1)

for key, val in dd.items():
    if hasattr(val, 'shape'):
        print(key, val.shape)
    else:
        print(key, type(val))

# y: mean on screen
# y1: mean on bpm1
# y2: mean on bpm2 # Actually they are identical :(

#deltaI: yI/disp

# Image [i,j] : i: n_shot j: n_gap

gap_list = dd['gap'].squeeze()*1e-3
n_shots = dd['nshots'].squeeze()

bpm1 = str(dd['BPM1'].squeeze())
bpm2 = str(dd['BPM2'].squeeze())


subplot = ms.subplot_factory(2, 3)
sp_ctr = 1

ms.figure('Undulator wakefield measurements')


xlabel = 'Undulator gap [mm]'
ylabel = 'Centroid position'
sp0 = subplot(sp_ctr, title='Normalized with dispersion', xlabel=xlabel, ylabel=r'Energy loss [MeV]')
sp_ctr += 1

sp_charge = subplot(sp_ctr, title='Charge profile', xlabel=r's [$\mu$m]', ylabel='Current (arb. units)')
sp_ctr += 1

bl_meas_file = data_dir + 'Bunch_length_meas_2020-02-03_21-54-24.h5'
total_charge = 200e-12

bl_meas = data_loader.load_blmeas(bl_meas_file)
current_profile = bl_meas['current1']

charge_profile = current_profile * total_charge / np.sum(current_profile)
charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
energy_eV = bl_meas['energy_eV']

sp_charge.plot(charge_xx*1e6, charge_profile)


for title, data_str, delta_str in zip(['BPM', 'Screen'], ['y1', 'y'], ['delta1', 'delta']):
    sp = subplot(sp_ctr, title=title, xlabel=xlabel, ylabel=ylabel)
    sp_ctr += 1

    xx = gap_list
    yy0 = dd[data_str].squeeze()
    if title == 'Screen':
        yy0 = yy0.T / 1e3
    yy = yy0.mean(axis=1)
    yy -= yy[0]
    yy_err = yy0.std(axis=1)
    sp.errorbar(xx*1e3, yy, yerr=yy_err)

    yy0 = dd[delta_str].squeeze() * energy
    if title == 'Screen':
        yy0 = yy0.T
    yy = yy0.mean(axis=1)
    yy -= yy[0]
    yy_err = yy0.std(axis=1)
    sp0.errorbar(xx*1e3, yy/1e6, yerr=yy_err/1e6, label=title)


#for gap in gap_list:
#
#    wf_dict = uwf.calc_all(charge_xx, charge_profile, gap/2., 48)
#    loss_surface = wf_dict['proj_Eloss_surface']
#    loss_ac = wf_dict['proj_Eloss_ac']
#
#    print(gap, loss_surface, loss_ac)


#sp_ctr = np.inf
#
#for n_gap, gap in enumerate(gap_list):
#    for n_shot in range(1):
#        if sp_ctr > 6:
#            ms.figure('Selected images')
#            sp_ctr = 1
#        image = dd['Image'][n_shot, n_gap].astype(np.float64)
#        x_axis = dd['x_axis'].squeeze()
#        y_axis = dd['y_axis'].squeeze()
#
#        sp = subplot(sp_ctr, title='Image %.2f %i' % (n_gap, n_shot), grid=False)
#        sp.imshow(image, aspect='auto', extent=(x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]))
#        sp_ctr += 1
#
#
#sp0.legend()



plt.show()

