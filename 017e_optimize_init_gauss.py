#import itertools
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
smoothen = 0e-6
n_emittances = (2000e-9, 2000e-9)
n_particles = int(100e3)
forward_method = 'matrix'
show_module = False
self_consistent = False


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

plot_ctr = 5
ny, nx = 3, 3
subplot = ms.subplot_factory(ny,nx)
sp_ctr = ny*nx+1

nfev_ctr = 0
max_nfev = 15

opt_plot = True


def opt_func(sig_t_fs, count_nfev, profile_cutoff, screen_cutoff, smoothen):
    a = np.array(opt_func_values)
    if len(a) > 0 and np.any(a[:,0] == sig_t_fs):
        index = np.argwhere(sig_t_fs == a[:,0])[0]
        return a[index, 1]

    global opt_ctr, sp_ctr, plot_ctr, sp, nfev_ctr, sp2

    sig_t = sig_t_fs/1e15

    bp_wake = tracking.get_gaussian_profile(sig_t, tt_halfrange, len_profile, charge, tracker.energy_eV)
    baf = tracker.back_and_forward(meas_screen, bp_wake, gaps, beam_offsets, n_streaker)

    if self_consistent:

        baf_self = tracker.back_and_forward(meas_screen, baf['beam_profile'], gaps, beam_offsets, n_streaker)
        screen_self = baf_self['screen']

    else:
        screen_self = baf['screen']
    profile = baf['beam_profile']

    diff = screen_self.compare(meas_screen)

    print(opt_ctr, '%f fs' % sig_t_fs, '%.1e' % diff)
    opt_ctr += 1

    if opt_plot:
        if plot_ctr == 5:
            plot_ctr = 0
            if sp_ctr == ny*nx+1:
                ms.figure('Optimization')
                sp_ctr = 1
            sp = subplot(sp_ctr, title='Screen')
            sp_ctr += 1
            sp.plot(meas_screen_shift.x*1e3, meas_screen_shift.intensity/meas_screen_shift.integral, label='Original')
            sp2 = subplot(sp_ctr, title='Profile')
            sp_ctr += 1
            sp2.plot(profile_meas.time*1e15, profile_meas.current/profile_meas.integral, label='Original')

        plot_ctr += 1
        sp.plot(screen_self.x*1e3, screen_self.intensity/screen_self.integral, label='%i: %.1f fs %.3e' % (opt_ctr, sig_t_fs, diff))
        sp2.plot(profile.time*1e15, profile.current/profile.integral, label='%i: %.1f fs %.3e' % (opt_ctr, sig_t_fs, diff))
        sp.legend()
        sp2.legend()
        plt.show()
        plt.pause(0.01)

    if count_nfev:
        nfev_ctr += 1
        if nfev_ctr > max_nfev:
            raise StopIteration(sig_t_fs)

    opt_func_values.append((float(sig_t_fs), diff))
    opt_func_screens.append(screen_self)
    opt_func_profiles.append(baf['beam_profile'])

    return diff


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

if show_module:

    sp_profile4 = subplot(sp_ctr2, title='Reconstructed profile using Module', xlabel='t [fs]', ylabel='Intensity (arb. units)')
    sp_ctr2 += 1
else:
    sp_profile4 = None

sp_screen3 = subplot(sp_ctr2, title='Reconstructed screen using self-consistent', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr2 += 1

if show_module:
    sp_screen4 = subplot(sp_ctr2, title='Reconstructed screen using Module', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr2 += 1
else:
    sp_screen4 = None

#sp_screen3 = sp_screen4 = None



compensate_negative_screen = True
for n_loop, (n_emittances) in enumerate(([10e-9, 10e-9], [500e-9, 500-9], [2000e-9, 2000e-9])):

    label = '%i' % (n_emittances[0]*1e9)

    tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, screen_bins=screen_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff, len_screen=len_profile, forward_method=forward_method, compensate_negative_screen=compensate_negative_screen)
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
    #meas_screen0 = fab_dict_real['track_dict_forward0']['screen']

    if n_loop == 0:
        for sp_ in sp_profile, sp_profile2, sp_profile3, sp_profile4:
            if sp_ is None:
                continue
            sp_.plot(profile_meas.time*1e15, profile_meas.current/profile_meas.integral, label='Real %i' % (profile_meas_sigma*1e15), lw=real_lw)

        for sp_ in sp_screen, sp_screen2, sp_screen3, sp_screen4:
            if sp_ is None:
                continue
            meas_screen_shift = meas_screen
            sp_.plot(meas_screen_shift.x*1e3, meas_screen_shift.intensity/meas_screen_shift.integral, label='Real', lw=real_lw)

    # Using gaussian wake profile
    opt_ctr = 0
    opt_func_values = []
    opt_func_screens = []
    opt_func_profiles = []


    sig_t_fs_arr = np.arange(25, 55.01, 5)
    diff_arr = np.array([opt_func(t, False, profile_cutoff, screen_cutoff, smoothen) for t in sig_t_fs_arr])
    index = np.argmin(diff_arr)
    sig_t_fs_min = sig_t_fs_arr[index]

    sig_t_fs_range2 = list(range(int(round(sig_t_fs_min-4)), int(round(sig_t_fs_min+4))))
    for sig_t_fs in sig_t_fs_range2:
        opt_func(sig_t_fs, False, profile_cutoff, screen_cutoff, smoothen)

    opt_func_values = np.array(opt_func_values)
    best_index = np.argmin(opt_func_values[:,1])
    best_sig_t = opt_func_values[best_index, 0]
    best_screen = opt_func_screens[best_index]
    best_profile = opt_func_profiles[best_index]

    sp_profile.plot(best_profile.time*1e15, best_profile.current/best_profile.integral, label=label+' %i' % (best_profile.gaussfit.sigma*1e15))
    sp_opt.scatter(opt_func_values[:,0], opt_func_values[:,1], label=label)

    # Using best gaussian recon for final step
    baf_dict_final = tracker.back_and_forward(meas_screen, best_profile, gaps, beam_offsets, n_streaker)
    screen_final = baf_dict_final['screen']
    #screen_final.shift()
    screen_final.cutoff(screen_cutoff)
    screen_final.normalize()
    profile_final = baf_dict_final['beam_profile']
    sp_screen3.plot(screen_final.x*1e3, screen_final.intensity, label=label)
    sp_profile3.plot((profile_final.time-profile_final.gaussfit.mean)*1e15, profile_final.current/profile_final.integral, label=label+' %i' % (profile_final.gaussfit.sigma*1e15))


    # Module code
    if show_module:
        sig_t_range = np.array(sig_t_fs_range2)*1e-15
        module_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True)
        profile_module = module_dict['reconstructed_profile']
        screen_module = module_dict['reconstructed_screen']
        sp_screen4.plot(screen_module.x*1e3, screen_module.intensity, label=label)
        sp_profile4.plot((profile_module.time-profile_module.gaussfit.mean)*1e15, profile_module.current/profile_module.integral, label=label+' %i' % (profile_module.gaussfit.sigma*1e15))

        iter_dict = tracker.iterate(meas_screen, best_profile, gaps, beam_offsets, n_streaker, 5)
        for n_iter, (screen, profile, diff) in enumerate(zip(iter_dict['screens'], iter_dict['profiles'], iter_dict['diffs'])):
            profile.center()
            screen.plot_standard(sp_screen4, label=n_iter)
            profile.plot_standard(sp_profile4, True, label=n_iter)


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

    sp_profile2.plot(profile_real.time*1e15, profile_real.current/profile_real.integral, label=label+' %i' % (profile_real.gaussfit.sigma*1e15))

    sp_screen2.plot(screen_shift.x*1e3, screen_shift.intensity/screen_shift.integral, label=label)
    sp_screen.plot(best_screen.x*1e3, best_screen.intensity/best_screen.integral, label=label)

    ms.figure('Investigation emittance %i' % (n_emittances[0]*1e9))
    sp_ctr = 1
    screen = copy.deepcopy(meas_screen)
    screen.smoothen(30e-6)
    sp_p = subplot(sp_ctr, title='Profiles')
    sp_ctr += 1
    sp_s = subplot(sp_ctr, title='Screens')
    sp_s.plot(screen.x*1e3, screen.intensity/screen.integral, color='black', label='Real')
    sp_ctr += 1
    #sp_b = subplot(3, title='Back agaun')
    r12 = tracker.calcR12()[0]
    for profile, label in [(best_profile, 'Gaussian'), (profile_final, 'Final self-consistent'), (profile_meas, 'Measured')]:
        #profile.shift()
        track_dict = forward_fun(profile, gaps, beam_offsets)
        screen_forward = track_dict['screen']
        #screen.cutoff(0.05)

        wf_dict = profile.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker])
        wake_effect = profile.wake_effect_on_screen(wf_dict, r12)
        bp_back = tracker.track_backward(screen, wake_effect)

        color = sp_p.plot((profile.time-profile.gaussfit.mean)*1e15, profile.current/profile.integral, label=label+' %i fs' % (profile.gaussfit.sigma*1e15))[0].get_color()
        sp_p.plot((bp_back.time-bp_back.gaussfit.mean)*1e15, bp_back.current/bp_back.integral, ls='--', label=label+' back'+' %i fs' % (bp_back.gaussfit.sigma*1e15))
        sp_s.plot(screen_forward.x*1e3, screen_forward.intensity/screen_forward.integral, label=label, color=color)
    sp_p.legend()
    sp_s.legend()

for sp_ in sp_profile, sp_opt, sp_profile2, sp_profile3, sp_profile4, sp_screen, sp_screen2, sp_screen3, sp_screen4:
    if sp_ is None:
        continue

    sp_.legend(title='Emittance')

#ms.saveall('./group_metting_2020-11-17/opt_gauss', hspace=0.4, vspace=0.3)

plt.show()

