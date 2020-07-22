import os
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from scipy.io import loadmat

import data_loader
from EmittanceTool.h5_storage import saveH5Recursive
import uwf_model as uwf
import myplotstyle as ms

plt.close('all')

# initial blmeas: Bunch_length_meas_2020-02-03_15-51-34.h5
# repeated blmeas: Bunch_length_meas_2020-02-03_21-54-24.h5
# indeed they look very similar
# -- uwf measurements for this spot:
# Eloss_UNDbis.mat
# Eloss_UND2ndStreak.mat
# Eloss_DEH1-COMP1.mat

# second setting, there are two measurements:
#   - Bunch_length_meas_2020-02-03_22-07-39.h5
#   - Bunch_length_meas_2020-02-03_22-08-38.h5
# -- uwf measurements  Eloss_DEH1-COMP2.mat Eloss_DEH1-COMP2.mat

# third setting (100 pC)
# Bunch_length_meas_2020-02-03_23-55-33.h5
# -- Eloss_UND-COMP3.mat Eloss_DEH1-COMP3.mat

data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
#data_dir = '/storage/data_2020-02-03/'
bl_meas_file = data_dir + 'Bunch_length_meas_2020-02-03_21-54-24.h5'
total_charge = 200e-12
trim = True

mat1 = data_dir + 'Eloss_UNDbis.mat'
dd = loadmat(mat1)

bl_meas = data_loader.load_blmeas(bl_meas_file)
current_profile = bl_meas['current1']

charge_profile = current_profile * total_charge / np.sum(current_profile)
charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
energy_eV = bl_meas['energy_eV']

if trim:
    mask = np.logical_and(charge_xx > 20e-6, charge_xx < 80e-6)
    charge_xx = charge_xx[mask]
    charge_profile = charge_profile1 = charge_profile[mask]
    charge_xx -= charge_xx.min()

charge_profile2 = np.zeros_like(charge_profile)
charge_profile2[charge_xx < 40e-6] = 1
charge_profile2 = charge_profile2 / np.sum(charge_profile2) * total_charge


charge_profile3 = np.zeros_like(charge_profile)
charge_profile3[charge_xx < 20e-6] = 1
charge_profile3 = charge_profile3 / np.sum(charge_profile3) * total_charge

for c_ctr, charge_profile in enumerate([charge_profile1, charge_profile2, charge_profile3]):


    gap_list = dd['gap'].squeeze()*1e-3

    def do_calc(gap):
        return uwf.calc_all(charge_xx, charge_profile, gap/2., L=48.)

    with Pool(cpu_count()) as p:
        result_list = p.map(do_calc, gap_list)

    result_dict = {str(i): result_list[i] for i, gap in enumerate(gap_list)}
    result_dict['gap_list'] = gap_list
    result_dict['charge_profile'] = charge_profile
    result_dict['charge_xx'] = charge_xx
    result_dict['energy_eV'] = energy_eV
    save_file = os.path.basename(mat1) + '_wake%i.h5' % c_ctr
    saveH5Recursive(save_file, result_dict)
sys.exit()


subplot = ms.subplot_factory(2,3)
sp_ctr = 1

sp_charge = subplot(sp_ctr, title='Current_profile')
sp_ctr += 1
sp_charge.plot(charge_xx*1e6, charge_profile)

xlabel = 's [$\mu$m]'
ylabel = 'w [kV/(pC$\cdot$m)]'
ylabel2 = '$\Delta$ E [MeV]'

sp_wf_surface = subplot(sp_ctr, title='Surface wake', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_wf_res = subplot(sp_ctr, title='Resistive wake', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_wf_surface_W = subplot(sp_ctr, title='Surface wake convolved', xlabel=xlabel, ylabel=ylabel2)
sp_ctr += 1

sp_wf_res_W = subplot(sp_ctr, title='Resistive wake convolved', xlabel=xlabel, ylabel=ylabel2)
sp_ctr += 1

sp_gap_effect = subplot(sp_ctr, title='Proj Energy loss', xlabel='Gap [mm]', ylabel='Energy loss [MeV]')
sp_ctr += 1

eloss_surface = []
eloss_ac = []

for gap, wf_dict in zip(gap_list, result_list):
    loss_surface = wf_dict['proj_Eloss_surface']
    loss_ac = wf_dict['proj_Eloss_ac']
    eloss_surface.append(loss_surface)
    eloss_ac.append(loss_ac)

    label = '%.1f' % (gap*1e3)

    plot_xx = charge_xx * 1e6
    factor1 = 1e-3*1e-12
    factor2 = 1e-6

    sp_wf_surface.errorbar(plot_xx, wf_dict['w_surface']*factor1, label=label, yerr=wf_dict['w_surface_err']*factor1)
    sp_wf_res.plot(plot_xx, wf_dict['w_ac']*factor1, label=label)
    sp_wf_surface_W.plot(plot_xx, wf_dict['W_surface']*factor2, label=label)
    sp_wf_res_W.plot(plot_xx, wf_dict['W_ac']*factor2, label=label)

sp_gap_effect.plot(gap_list, eloss_surface, label='Surface')
sp_gap_effect.plot(gap_list, eloss_ac, label='Resistive')

sp_gap_effect.legend()

for sp_ in sp_wf_res, sp_wf_surface, sp_wf_res_W, sp_wf_surface_W:
    sp_.legend()

for gap, wf_dict in zip(gap_list, result_list):
    print(gap, loss_surface, loss_ac)

plt.show()

