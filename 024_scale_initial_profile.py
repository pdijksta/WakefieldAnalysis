#import itertools
import copy; copy
import socket
import numpy as np; np
from scipy.constants import c
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

gaps = [10e-3, 10e-3]
beam_offsets = [4.75e-3, 0]
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 100e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.00
profile_cutoff = 0.00
len_profile = 5e3
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = (300e-9, 300e-9)
n_particles = int(100e3)

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, n_particles=n_particles, screen_cutoff=screen_cutoff, screen_bins=screen_bins, smoothen=smoothen, profile_cutoff=profile_cutoff, len_screen=len_profile)
energy_eV = tracker.energy_eV

scale_factors = np.array(list(np.exp(np.linspace(np.log(0.05), np.log(0.7), 9)))+[1])
sig_t_range = np.exp(np.linspace(np.log(1), np.log(45), 10))*1e-15
#sig_t_range2 = np.arange(-6, 6.01, 1)*1e-15


profile_meas0 = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
profile_meas0.center()


real_profiles, opt_profiles = [], []
real_screens, opt_screens = [], []

for scale_factor in scale_factors:


    profile_meas = copy.deepcopy(profile_meas0)
    profile_meas.scale_xx(scale_factor)
    profile_meas.center()

    profile_meas.scale_yy(scale_factor)
    new_charge = charge*scale_factor
    profile_meas.charge = new_charge
    #print('%.2f %i %i %i' % (scale_factor, new_charge*1e12, profile_meas.gaussfit.sigma*1e15, profile_meas.current.sum()*1e12))



    ms.figure('Scale factor %.2f, %i pC' % (scale_factor, new_charge*1e12))
    subplot = ms.subplot_factory(2,3)
    sp_ctr = 1

    new_tt_halfrange = (profile_meas.time.max() - profile_meas.time.min())/2.

    meas_screen = tracker.matrix_forward(profile_meas, gaps, beam_offsets)['screen']
    real_screens.append(meas_screen)

    screen0 = tracker.matrix_forward(profile_meas, gaps, [0,0])['screen']

    opt_dict = tracker.find_best_gauss(sig_t_range, new_tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, new_charge)
    sig0 = opt_dict['gauss_sigma']

    scale_range = scale_factors*1.2

    scale_profile0 = copy.deepcopy(profile_meas0)
    scale_profile0.scale_yy(scale_factor)

    opt_dict_scale = tracker.scale_existing_profile(scale_range, scale_profile0, meas_screen, gaps, beam_offsets, n_streaker)

    scale_profile = opt_dict_scale['reconstructed_profile']
    scale_profile.center()
    scale_screen = opt_dict_scale['reconstructed_screen']

    opt_profile = opt_dict['reconstructed_profile']
    opt_profile.center()
    opt_screen = opt_dict['reconstructed_screen']
    opt_screens.append(opt_screen)

    if False:
        scaled_profiles = opt_dict_scale['all_profiles']
        scaled_profiles0 = opt_dict_scale['scaled_profiles']
        scaled_screens = opt_dict_scale['all_screens']
        sp_debug_p = subplot(sp_ctr, title='Debug profiles')
        sp_ctr += 1

        profile_meas.plot_standard(sp_debug_p, color='black', lw=3, norm=True)
        sp_debug_s = subplot(sp_ctr, title='Debug screens')

        meas_screen.plot_standard(sp_debug_s, color='black', lw=3)
        sp_ctr += 1

        for n_scale, (opt_scale, p, p2, s) in enumerate(zip(scale_range, scaled_profiles0, scaled_profiles, scaled_screens)):
            p.center()
            p2.center()

            color = p.plot_standard(sp_debug_p, norm=True, label='%i %.2f' % (n_scale, opt_scale))[0].get_color()
            p2.plot_standard(sp_debug_p, norm=True, color=color, ls='--')
            s.plot_standard(sp_debug_s, label='%i %.2f' % (n_scale, opt_scale))

        sp_debug_p.legend()
        sp_debug_s.legend()


    real_profiles.append(profile_meas)
    opt_profiles.append(opt_profile)




    sp_profile = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
    sp_ctr += 1

    sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Screen distribution (arb. units)')
    sp_ctr += 1

    sp_wake = subplot(sp_ctr, title='Wakefield', xlabel='t [fs]', ylabel='V/m')
    sp_ctr += 1

    for profile, label in [(profile_meas, 'Real'), (opt_profile, 'Reconstructed')]:
        wake_dict = profile.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker])
        wake = wake_dict['dipole']['wake_potential']
        wake_tt = wake_dict['input']['charge_xx']/c
        sp_wake.plot(wake_tt, wake, label=label)

    sp_wake.legend()

    profile_meas.plot_standard(sp_profile, norm=True, color='black', lw=3, label='R %.1f' % (profile_meas.gaussfit.sigma*1e15))
    opt_profile.plot_standard(sp_profile, norm=True, label='G %.1f' % (opt_profile.gaussfit.sigma*1e15))
    scale_profile.plot_standard(sp_profile, norm=True, label='S %.1f' % (scale_profile.gaussfit.sigma*1e15))

    meas_screen.plot_standard(sp_screen, color='black', lw=3, label='R')
    opt_screen.plot_standard(sp_screen, label='G')
    scale_screen.plot_standard(sp_screen, label='S')

    sp_profile.legend()
    sp_screen.legend()


#print(real_profiles[0].charge)

ms.figure('Scaling summary')
ny, nx = 2, 3
subplot = ms.subplot_factory(ny,nx)
sp_ctr = 1

plot_ctr = np.inf
for real_profile, opt_profile, scale_factor, real_screen, opt_screen in zip(real_profiles, opt_profiles, scale_factors, real_screens, opt_screens):
    if plot_ctr > 3:
        sp_profile = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
        sp_ctr += 1
        plot_ctr = 0

        sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Screen distribution (arb. units)')
        sp_ctr += 1

    label = '%i / %i' % (real_profile.gaussfit.sigma*1e15, real_profile.charge*1e12)
    color = real_profile.plot_standard(sp_profile, label=label, norm=True)[0].get_color()
    opt_profile.plot_standard(sp_profile, color=color, ls='--', norm=True)

    real_screen.plot_standard(sp_screen, label=label, color=color)
    opt_screen.plot_standard(sp_screen, ls='--', color=color)





    sp_profile.legend(title='Duration [fs] / Charge [pC]')
    plot_ctr += 1


plt.show()

