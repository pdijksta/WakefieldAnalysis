import numpy as np; np
from scipy.constants import c
import matplotlib.pyplot as plt
from scipy.io import loadmat
import myplotstyle as ms

import wf_model
import data_loader


data_dir = '/storage/data_2020-02-03/'
data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
data_dir = '/sf/data/measurements//2020/02/03/'
bl_meas_file = data_dir + 'Bunch_length_meas_2020-02-03_15-59-13.h5'
bl_meas = data_loader.load_blmeas(bl_meas_file)
total_charge = 200e-12

current_profile = bl_meas['current1']
charge_profile = current_profile * total_charge / np.sum(current_profile)
charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
energy_eV = bl_meas['energy_eV']


plt.close('all')

files = [data_dir+'/Eloss_DEH1.fig.mat', data_dir+'/Eloss_DEH2.fig.mat', data_dir+'/Eloss_DEH1-2.fig.mat']
lengths = [1, 1, 2]
titles = ['Structure 1', 'Structure 2', 'Structures 1+2']

for n_struct, (file_, Ls, title) in enumerate(zip(files, lengths, titles)):
    print(title)
    mat = loadmat(file_)
    wf_calc = wf_model.WakeFieldCalculator(charge_xx, charge_profile, energy_eV, Ls=Ls)

    name_dict = {
            'Energy_spread': {
                'Gaussian': 'Gaussian_fit',
                'FWHM': 'direct_FWHM',
            },
            'Energy_loss': {
                'BPM': 'BPM1',
                'Screen': 'screen',
                },
            }

    data = {}
    for key1, subdict in name_dict.items():
        data[key1] = {}
        for key2, alternate_key in subdict.items():
            data[key1][key2] = {
                    'xx': mat[alternate_key+'_XData'].squeeze(),
                    'yy': mat[alternate_key+'_YData'].squeeze(),
                    'err1': mat[alternate_key+'_YPositiveDelta'].squeeze(),
                    'err2': mat[alternate_key+'_YNegativeDelta'].squeeze(),
                    }

    full_title = 'Longitudinal measurements %s' % title
    fig = ms.figure(title=full_title)

    subplot = ms.subplot_factory(2, 3)
    sp_ctr = 1


    # Plot Raw_data
    sp_eloss = subplot(sp_ctr, title='Energy loss', xlabel='Gap [mm]', ylabel='Rel. energy loss', sciy=True)
    sp_ctr += 1

    sp_espread = subplot(sp_ctr, title='Energy spread', xlabel='Gap [mm]', ylabel='Rel. energy spread', sciy=True)
    sp_ctr += 1

    sp_charge_profile = subplot(sp_ctr, title='Charge profile', xlabel=r's [$\mu$m]', ylabel=r'Charge profile [pC / $\mu$m]')
    sp_ctr += 1

    xx_um = wf_calc.xx*1e6
    sp_charge_profile.plot(xx_um, wf_calc.charge_profile*1e12/np.diff(xx_um)[0])

    sp_wake_functions = subplot(sp_ctr, title='Single particle wake functions', xlabel=r's [$\mu$m]', ylabel='E [MV/(m nC)')
    sp_ctr += 1

    sp_wake_potentials = subplot(sp_ctr, title='Wake potentials', xlabel=r's [$\mu$m]', ylabel='E [MV/(m nC)')
    sp_ctr += 1

    for key, data_dict in data['Energy_loss'].items():
        yy = data_dict['yy']
        sp_eloss.errorbar(data_dict['xx'], yy-yy[0], yerr=data_dict['err1'], label=key)


    for key, data_dict in data['Energy_spread'].items():
        yy = data_dict['yy']
        sp_espread.errorbar(data_dict['xx'], yy-yy[0], yerr=data_dict['err1'], label=key)


    # Plot model data
    gap_list = data_dict['xx']
    eloss_model_list = []
    espread_model_list = []

    for gap_mm in gap_list:
        semigap_m = gap_mm/2*1e-3
        eloss_dict = wf_calc.calc_all(semigap_m, 1, beam_offset=0, calc_lin_dipole=False, calc_dipole=False, calc_quadrupole=False, calc_long_dipole=True)['longitudinal_dipole']

        rel_eloss = eloss_dict['mean_energy_loss']/energy_eV
        rel_espread = eloss_dict['espread_increase']/energy_eV
        eloss_model_list.append(rel_eloss)
        espread_model_list.append(rel_espread)
        #print(gap_mm, '%.1e' % rel_eloss)
        sp_wake_functions.plot(wf_calc.xx*1e6, eloss_dict['single_particle_wake']*1e-15, label='%.1f' % gap_mm)
        sp_wake_potentials.plot(wf_calc.xx*1e6, eloss_dict['wake_potential']*1e-15, label='%.1f' % gap_mm)

    eloss_model_list = np.array(eloss_model_list)
    espread_model_list = np.array(espread_model_list)

    sp_eloss.plot(gap_list, -(eloss_model_list-eloss_model_list[0]), label='Model')
    yy_model = espread_model_list-espread_model_list[0]
    sp_espread.plot(gap_list, yy_model, label='Model')

    sp_espread.set_ylim(-1e-4, yy_model.max()*1.2)

    sp_wake_functions.legend(title='Gap [mm]')
    sp_eloss.legend(loc='upper left')
    sp_espread.legend(loc='upper right')



ms.saveall('~/Dropbox/plots/006_longitudinal_measurements', ending='.pdf', bottom=0.15, wspace=0.3, top=0.85)



plt.show()

