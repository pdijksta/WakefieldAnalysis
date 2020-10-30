import copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import curve_fit

import myplotstyle as ms

plt.close('all')


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

#energy_eV = 6.14e9
gaps = [10e-3, 10e-3]
beam_offsets = [4.7e-3, 0.]
fit_order = 4
sig_t = 30e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
backtrack_cutoff = 0.05
len_profile = 1e3

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, energy_eV='file')
energy_eV = tracker.energy_eV

profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
#profile_meas.reshape(1e5)
profile_gauss_guess = tracking.get_gaussian_profile(sig_t, tt_halfrange, len_profile, charge, energy_eV)

for profile in profile_meas, profile_gauss_guess:
    profile.calc_wake(gaps[0], beam_offsets[0], 1.)

ms.figure('Forward and backward tracking - Ignore natural beamsize (200 nm emittance)')
subplot = ms.subplot_factory(2,3)
sp_ctr = 1

sp0 = subplot(sp_ctr, title='Current profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp_f = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Screen distribution')
sp_ctr += 1

track_dict0 = tracker.elegant_forward(profile_meas, gaps, [0., 0.], [1., 1.])
r12 = track_dict0['r12_dict'][0]

track_dict_streak = tracker.elegant_forward(profile_meas, gaps, beam_offsets, [1., 1.])
track_dict_guess = tracker.elegant_forward(profile_gauss_guess, gaps, beam_offsets, [1., 1.])


sp = subplot(sp_ctr, title='Guessed wake effect', xlabel='t [fs]', ylabel='x [mm]')
sp_ctr += 1

for profile, label in [(profile_meas, 'Measured'), (profile_gauss_guess, 'Initial guess')]:
    wf_dict = profile.calc_wake(gaps[0], beam_offsets[0], 1.)
    wake_effect = profile.wake_effect_on_screen(wf_dict, r12)['x']
    sp.plot(profile.time*1e15, wake_effect*1e3, label=label)
    #sp1.plot(profile.time*1e15, wf_dict['dipole']['wake_potential'], label=label)

sp.legend()


# Backtrack
wf_dict = profile_gauss_guess.calc_wake(gaps[0], beam_offsets[0], 1.)
wake_effect = profile_gauss_guess.wake_effect_on_screen(wf_dict, r12)
profile_bt0 = tracker.track_backward(track_dict_streak, track_dict0, wake_effect)
profile_bt0.reshape(len_profile)
profile_bt_cut = copy.deepcopy(profile_bt0)
profile_bt_cut.cutoff(backtrack_cutoff)

for bp, label in [
        (profile_meas, 'Measured'),
        (profile_gauss_guess, 'Initial guess'),
        (profile_bt0, 'Backtracked'),
        #(profile_bt_cut, 'Backtracked cut'),
        ]:
    label2 = label + ' %i fs' % (bp.gaussfit.sigma*1e15)
    norm = np.trapz(bp.current, bp.time*1e15)

    color = sp0.plot(bp.time*1e15, bp.current/norm, label=label2)[0].get_color()
    gfx, gfy = bp.gaussfit.xx, bp.gaussfit.reconstruction
    sp0.plot(gfx*1e15, gfy/norm, ls='--', color=color)
    if profile is profile_bt0:
        sp0.axhline(bp.current.max()/norm*backtrack_cutoff, ls='--', color='black', label='Cutoff')

    comp = profile_meas.compare(bp)
    print('Diff to %s: %.1e' % (label, comp))
sp0.legend()

track_dict_bt0 = tracker.elegant_forward(profile_bt0, gaps, beam_offsets, [1., 1.])
track_dict_bt_cut = tracker.elegant_forward(profile_bt0, gaps, beam_offsets, [1., 1.])

for track_dict, label in [
        (track_dict0, 'No streaking'),
        (track_dict_streak, 'Streaking'),
        (track_dict_guess, 'Initial guess'),
        (track_dict_bt0, 'Back and forward raw'),
        (track_dict_bt_cut, 'Back and forward cut'),
        ]:

    screen_x = track_dict['screen'].x
    screen_hist = track_dict['screen'].intensity
    norm = np.trapz(screen_hist, screen_x)
    if track_dict is track_dict0:
        sp_f.step(screen_x*1e3, screen_hist/norm/8, label=label, ls='--')
    else:
        sp_f.step(screen_x*1e3, screen_hist/norm, label=label)

sp_f.legend()

plt.show()

