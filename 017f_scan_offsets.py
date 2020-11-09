"""
Obsolete
"""
import functools
import socket
import numpy as np; np
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

#energy_eV = 6.14e9
gaps = [10e-3, 10e-3]
beam_offsets_s0 = [4.6e-3, 4.7e-3, 4.8e-3]
beam_offset_s1 = 0.
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
backtrack_cutoff = 0.05
len_profile = 1e3
struct_lengths = [1., 1.]
n_bins=150

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file')
energy_eV = tracker.energy_eV

profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
profile_gauss = tracking.get_gaussian_profile(sig_t, tt_halfrange, len_profile, charge, energy_eV)

for beam_offset0 in beam_offsets_s0:
    beam_offsets = [beam_offset0, beam_offset_s1]
    fab_dict_real = tracker.forward_and_back(profile_meas, profile_meas, gaps, beam_offsets, n_streaker, n_bins, output='Full')
    fab_dict_real['bp_back'].cutoff(0.1)
    #fab_dict_gauss = tracker.forward_and_back(profile_meas, profile_gauss, gaps, beam_offsets, n_streaker, output='Full')
    #fab_dict_gauss['bp_back'].cutoff(0.1)
    #forward_dict_real = tracker.elegant_forward(fab_dict_real['bp_back'], gaps, beam_offsets)
    #forward_dict_gauss = tracker.elegant_forward(fab_dict_gauss['bp_back'], gaps, beam_offsets)

    #profile_back_real = fab_dict_real['bp_back']
    #profile_back_gauss = fab_dict_gauss['bp_back']
    meas_screen = fab_dict_real['track_dict_forward']['screen']
    meas_screen0 = fab_dict_real['track_dict_forward0']['screen']

    meas_screen2 = tracker.back_and_forward(meas_screen, meas_screen0, profile_meas, gaps, beam_offsets, n_streaker, n_bins, back_cutoff=0.1)
    meas_screen3 = tracker.back_and_forward(meas_screen, meas_screen0, profile_gauss, gaps, beam_offsets, n_streaker, n_bins, back_cutoff=0.1)

    ms.figure('Back and forward with real profile %.1e offset' % beam_offset0)

    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp = subplot(sp_ctr, title='Current profiles')
    sp_ctr += 1

    for bp, label in [
            (profile_meas, 'Measured'),
            #(profile_back_real, 'Reconstructed using real'),
            #(profile_back_gauss, 'Reconstructed using gauss'),
            ]:
        norm = np.trapz(bp.current, bp.time)
        sp.plot(bp.time*1e15, bp.current/norm, label=label)

    sp.legend()

    sp = subplot(sp_ctr, title='Screen distributions')
    sp_ctr += 1
    #meas_screen = tracker.elegant_forward(profile_meas, gaps, beam_offsets)['screen']
    #meas_screen0 = tracker.elegant_forward(profile_meas, gaps, [0, 0])['screen']

    for sd, label in [
            (meas_screen, 'Streaking'),
            (meas_screen0, 'No streaking'),
            (meas_screen2, 'Streaking back and forward real'),
            (meas_screen3, 'Streaking back and forward gauss'),
            #(forward_dict_gauss['screen'], 'Forward gauss'),
            #(forward_dict_real['screen'], 'Forward real'),
            ]:
        sp.plot(sd.x*1e3, sd.intensity, label=label)
    sp.legend()


    ctr = 0

    plot_ctr = 5
    sp_ctr = 7
    subplot = ms.subplot_factory(2,3)

    meas_screen_max_x = meas_screen.x[np.argmax(meas_screen.intensity)]
    meas_screen_shift = tracking.ScreenDistribution(meas_screen.x-meas_screen_max_x, meas_screen.intensity.copy())
    meas_screen_shift.cutoff(0.1)

    nfev_ctr = 0
    max_nfev = 15
    opt_func_values = []

    @functools.lru_cache()
    def opt_func(sig_t_fs, count_nfev):
        global ctr, sp_ctr, plot_ctr, sp, nfev_ctr

        sig_t = sig_t_fs/1e15

        bp_wake = tracking.get_gaussian_profile(sig_t, tt_halfrange, len_profile, charge, tracker.energy_eV)
        screen_recon = tracker.back_and_forward(meas_screen, meas_screen0, bp_wake, gaps, beam_offsets, n_streaker, n_bins, back_cutoff=0.1)
        screen_max_x = screen_recon.x[np.argmax(screen_recon.intensity)]
        screen_shift = tracking.ScreenDistribution(screen_recon.x-screen_max_x, screen_recon.intensity.copy())
        screen_shift.cutoff(0.1)

        diff = screen_shift.compare(meas_screen_shift)

        print(ctr, '%f fs' % sig_t_fs, '%.1e' % diff)
        ctr += 1

        if plot_ctr == 5:
            plot_ctr = 0
            if sp_ctr == 7:
                ms.figure('Optimization bo %.1e' % beam_offset0)
                sp_ctr = 1
            sp = subplot(sp_ctr, title='Profile')
            sp_ctr += 1
            sp.plot(meas_screen_shift.x*1e3, meas_screen_shift.intensity, label='Original')

        plot_ctr += 1
        sp.plot(screen_shift.x*1e3, screen_shift.intensity, label='%i: %.1f fs %.3e' % (ctr, sig_t_fs, diff))
        sp.legend()
        plt.show()
        plt.pause(0.01)

        if count_nfev:
            nfev_ctr += 1
            if nfev_ctr > max_nfev:
                raise StopIteration(sig_t_fs)

        opt_func_values.append((float(sig_t), diff))

        return diff

    sig_t_fs_arr = np.arange(10, 60.01, 5)
    diff_arr = np.array([opt_func(t, False) for t in sig_t_fs_arr])
    index = np.argmin(diff_arr)
    sig_t_fs_min = sig_t_fs_arr[index]

    #for sig_t_fs in range(int(sig_t_fs_min-4), int(sig_t_fs_min+4)):
    #    opt_func(sig_t_fs, False)

    #try:
    #    minimize(opt_func, sig_t_fs_min, args=(True,), options={'maxfev': 15, 'maxiter': 3, 'eps': 0.5}, method='CG')
    #except StopIteration:
    #    pass

    for sig_t_fs in range(int(sig_t_fs_min)-4, int(sig_t_fs_min)+4):
        opt_func(sig_t_fs, False)

    opt_func_values = np.array(opt_func_values)
    best_index = np.argmin(opt_func_values[:,1])
    best_sig_t = opt_func_values[best_index, 0]

    ms.figure('Scan results %.1e' % beam_offset0)
    sp_ctr = 1

    sp = subplot(sp_ctr, title='Optimization results', xlabel='Gaussian sigma [fs]', ylabel='Optimization func (arb. units)')
    sp_ctr += 1
    sp.scatter(opt_func_values[:,0]*1e15, opt_func_values[:,1])

    plt.show()

