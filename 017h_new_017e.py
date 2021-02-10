import itertools
import copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

gaps = [10e-3, 10e-3]
beam_offsets_true = [4.7e-3, 0]
offset_error = 00e-6
beam_offsets = [beam_offsets_true[0] + offset_error, beam_offsets_true[1]]
n_streaker = 0
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.00
profile_cutoff = 0.00
len_profile = 6e3
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 30e-6
n_emittances = (800e-9, 800e-9)
n_particles = int(10e3)
forward_method = 'matrix'
self_consistent = True
bp_smoothen = 2e-15
compensate_negative_screen = True
sig_t_range = np.arange(30, 60, 5)


if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
elif hostname == 'pubuntu':
    magnet_file = '/home/work/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/home/work/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'

opt_ctr = 0


ms.figure('Final forward')
subplot = ms.subplot_factory(2,2)
sp_ff = subplot(1, title='Final screen forward')

plot_ctr = 5
ny, nx = 3, 3
subplot = ms.subplot_factory(ny,nx)
sp_ctr = ny*nx+1

nfev_ctr = 0
max_nfev = 15

opt_plot = True

ms.figure('Scan results; offset error %i' % (offset_error*1e6), figsize=(16, 12))
sp_ctr2 = 1

real_lw=4

sp_opt = subplot(sp_ctr2, title='Optimization results', xlabel='Gaussian sigma [fs]', ylabel='Optimization func (arb. units)')
sp_ctr2 += 1

sp_profile = subplot(sp_ctr2, title='Reconstructed profile using Gauss', xlabel='t [fs]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

sp_screen = subplot(sp_ctr2, title='Reconstructed screen using Gauss', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

sp_profile2 = subplot(sp_ctr2, title='Reconstructed profile using real', xlabel='t [fs]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

sp_screen2 = subplot(sp_ctr2, title='Reconstructed screen using real', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

sp_profile3 = subplot(sp_ctr2, title='Reconstructed profile using self-consistent', xlabel='t [fs]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

sp_screen3 = subplot(sp_ctr2, title='Reconstructed screen using self-consistent', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1


for n_loop, (quad_wake, n_particles, bp_smoothen) in enumerate(itertools.product(
        [True],
        [20e3, 40e3, 80e3],
        [1e-15],
        )):

    label = 'n_particles %ik' % (n_particles//1e3)
    n_particles = int(n_particles)
    always_plot_real = False

    tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, screen_bins=screen_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff, len_screen=len_profile, forward_method=forward_method, compensate_negative_screen=compensate_negative_screen, quad_wake=quad_wake, bp_smoothen=bp_smoothen)
    energy_eV = tracker.energy_eV

    if forward_method == 'matrix':
        forward_fun = tracker.matrix_forward
    elif forward_method == 'elegant':
        forward_fun = tracker.elegant_forward

    profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
    profile_meas_sigma = profile_meas.gaussfit.sigma
    print(profile_meas.charge, profile_meas.current.sum())
    profile_meas.cutoff(profile_cutoff)
    profile_meas.reshape(len_profile)
    print(profile_meas.charge, profile_meas.current.sum())
    fab_dict_real = tracker.forward_and_back(profile_meas, profile_meas, gaps, beam_offsets_true, n_streaker)
    meas_screen = fab_dict_real['track_dict_forward']['screen']

    if always_plot_real or n_loop == 0:
        for sp_ in sp_profile, sp_profile2, sp_profile3:
            if sp_ is None:
                continue
            profile_meas.plot_standard(sp_, label='Real %i %s' % (profile_meas_sigma*1e15, label), lw=real_lw)

        for sp_ in sp_screen, sp_screen2, sp_screen3:
            if sp_ is None:
                continue
            meas_screen.plot_standard(sp_, label='Real', lw=real_lw)



    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent, details=True)

    best_profile = gauss_dict['reconstructed_profile']
    best_screen = gauss_dict['reconstructed_screen']
    opt_func_values = gauss_dict['opt_func_values']
    opt_func_screens = gauss_dict['opt_func_screens']
    opt_func_profiles = gauss_dict['opt_func_profiles']

    best_profile.plot_standard(sp_profile, label=label+' %i' % (best_profile.gaussfit.sigma*1e15))
    sp_opt.scatter(opt_func_values[:,0], opt_func_values[:,1], label=label)

    # Using best gaussian recon for final step
    baf_dict_final = tracker.back_and_forward(meas_screen, best_profile, gaps, beam_offsets, n_streaker)
    screen_final = baf_dict_final['screen']
    #screen_final.shift()
    screen_final.cutoff(screen_cutoff)
    screen_final.normalize()
    profile_final = baf_dict_final['beam_profile']
    screen_final.plot_standard(sp_screen3, label=label)
    profile_final.plot_standard(sp_profile3, label=label+' %i' % (profile_final.gaussfit.sigma*1e15))


    # Using real wake profile
    baf = tracker.back_and_forward(meas_screen, profile_meas, gaps, beam_offsets, n_streaker)
    profile_real = baf['beam_profile']
    screen_recon = baf['screen']
    #screen_max_x = screen_recon.x[np.argmax(screen_recon.intensity)]
    #screen_shift = tracking.ScreenDistribution(screen_recon.x-screen_max_x, screen_recon.intensity.copy())
    screen_shift = copy.deepcopy(screen_recon)
    #screen_shift.shift()
    screen_shift.cutoff(screen_cutoff)
    screen_shift.normalize()

    profile_real.plot_standard(sp_profile2, label=label+' %i' % (profile_real.gaussfit.sigma*1e15))

    screen_shift.plot_standard(sp_screen2, label=label)
    best_screen.plot_standard(sp_screen, label=label)

    ms.figure('Investigation %s' % label)
    sp_ctr = 1
    screen = copy.deepcopy(meas_screen)
    #screen.smoothen(30e-6)
    sp_p = subplot(sp_ctr, title='Profiles')
    sp_ctr += 1
    sp_s = subplot(sp_ctr, title='Screens')
    screen.plot_standard(sp_s, color='black', label='Real')
    sp_ctr += 1
    #sp_b = subplot(3, title='Back again')
    r12 = tracker.calcR12()[0]
    for profile, label2 in [(best_profile, 'Gaussian'), (profile_final, 'Final self-consistent'), (profile_meas, 'Measured')]:
        #profile.shift()
        track_dict = forward_fun(profile, gaps, beam_offsets)
        screen_forward = track_dict['screen']
        #screen.cutoff(0.05)

        wf_dict = profile.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker])
        wake_effect = profile.wake_effect_on_screen(wf_dict, r12)
        bp_back = tracker.track_backward(screen, wake_effect, n_streaker)

        color = profile.plot_standard(sp_p, label=label2+' %i fs' % (profile.gaussfit.sigma*1e15))[0].get_color()
        bp_back.plot_standard(sp_p, ls='--', label=label2+' back'+' %i fs' % (bp_back.gaussfit.sigma*1e15))
        screen_forward.plot_standard(sp_s, label=label2, color=color)


    ff = tracker.matrix_forward(profile_final, gaps, beam_offsets)
    screen_ff = ff['screen']
    meas_screen.plot_standard(sp_ff, label=label+' Ref')
    screen_ff.plot_standard(sp_ff, label=label)

    sp_p.legend()
    sp_s.legend()

sp_ff.legend()

for sp_ in sp_profile, sp_opt, sp_profile2, sp_profile3, sp_screen, sp_screen2, sp_screen3:
    if sp_ is None:
        continue

    sp_.legend(title='Emittance')

#ms.saveall('./group_metting_2020-11-17/opt_gauss', hspace=0.4, vspace=0.3)

plt.show()

