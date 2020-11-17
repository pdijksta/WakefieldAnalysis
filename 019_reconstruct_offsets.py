#import itertools
from collections import OrderedDict
import copy; copy
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
beam_offset_list = ([4.7e-3, 0], [4.75e-3, 0], [4.65e-3, 0])
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.02
profile_cutoff = 0
len_profile = 1e3
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


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, screen_bins=screen_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff, len_screen=len_profile)
energy_eV = tracker.energy_eV

profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
profile_meas.center()

meas_screen_dict = {}

ms.figure('Compare offset errors')
sp_ctr = 1
subplot = ms.subplot_factory(2,2)

sp_profile0 = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1
sp_profile0.plot(profile_meas.time*1e15, profile_meas.current/profile_meas.integral, label='Real / %i' % (profile_meas.gaussfit.sigma*1e15), color='black', lw=3)

offset_errors_s1 = (np.arange(-20, 20.01, 10)+0)*1e-6


sp_screen_dict = OrderedDict()
for n_bo, beam_offsets0 in enumerate(beam_offset_list):
    sp = subplot(sp_ctr, title='Measured screen dist offset %.2f mm' % (beam_offsets0[0]*1e3), xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_screen_dict[n_bo] = sp

for n_oe, offset_error_s1 in enumerate(offset_errors_s1):
    offset_error = np.array([offset_error_s1, 0])

    ms.figure('Different offsets (error=%.2f $\mu$m)' % (offset_error_s1*1e6))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_profile = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
    sp_ctr += 1

    sp_profile.plot(profile_meas.time*1e15, profile_meas.current/profile_meas.integral, label='Real', color='black', lw=3)

    sp_screen = subplot(sp_ctr, title='Measured screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    f_dict0 = tracker.matrix_forward(profile_meas, gaps, [0., 0.])
    meas_screen0 = f_dict0['screen']
    meas_screen0.normalize()

    meas_screen0 = tracker.matrix_forward(profile_meas, gaps, [0, 0])['screen']

    final_profile_list = []

    for n_bo, beam_offsets0 in enumerate(beam_offset_list):

        beam_offsets = np.array(beam_offsets0) + np.array(offset_error)
        f_dict_meas = tracker.matrix_forward(profile_meas, gaps, beam_offsets0)
        meas_screen = f_dict_meas['screen']
        meas_screen.normalize()
        meas_screen_dict[n_bo] = meas_screen
        if n_oe == 0:
            sp_screen_dict[n_bo].plot(meas_screen.x*1e3, meas_screen.intensity/meas_screen.integral, color='Black', label='Measured')



        label = '%.2f' % (beam_offsets[0]*1e3)
        color = sp_screen.plot(meas_screen.x*1e3, meas_screen.intensity, label=label)[0].get_color()

        sig_t_range = np.arange(20, 60.01, 5)*1e-15
        gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets, n_streaker, charge, self_consistent=False)
        sig_t_range2 = np.arange(-3, 3, 1)*1e-15 + gauss_dict['gauss_sigma']
        gauss_dict2 = tracker.find_best_gauss(sig_t_range2, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets, n_streaker, charge, self_consistent=False)

        reconstructed_screen = gauss_dict2['reconstructed_screen']
        reconstructed_screen.normalize()
        #sp_screen.plot(reconstructed_screen.x*1e3, reconstructed_screen.intensity, ls='-.', color=color)

        reconstructed_profile = gauss_dict2['reconstructed_profile']


        #bp = copy.deepcopy(reconstructed_profile)
        #sp_profile.plot(bp.time*1e15, bp.current/bp.integral, label=label, color=color)
        #final_profile_list.append(bp)


        final_baf = tracker.back_and_forward(meas_screen, reconstructed_profile, gaps, beam_offsets, n_streaker)
        final_bp = final_baf['beam_profile']
        final_screen = final_baf['screen']
        final_screen.normalize()
        final_profile_list.append(final_bp)
        final_bp.center()
        #final_bp.shift(profile_meas._xx[np.argmax(profile_meas._yy)])
        sp_profile.plot(final_bp.time*1e15, final_bp.current/final_bp.integral, color=color)
        sp_screen.plot(final_screen.x*1e3, final_screen.intensity, color=color, ls='--')


    xx, yy = tracking.get_average_profile(final_profile_list)
    avg_profile = tracking.BeamProfile(xx, yy, tracker.energy_eV, charge)
    #avg_profile.shift(profile_meas._xx[np.argmax(profile_meas._yy)])
    avg_profile.center()
    sp_profile.plot(avg_profile.time*1e15, avg_profile.current/avg_profile.integral, label='Average', color='red')
    sp_profile0.plot(avg_profile.time*1e15, avg_profile.current/avg_profile.integral, label='%i / %i' % (offset_error_s1*1e6, avg_profile.gaussfit.sigma*1e15))

    for n_bo, beam_offsets0 in enumerate(beam_offset_list):

        beam_offsets = np.array(beam_offsets0) + np.array(offset_error)
        f_avg = tracker.matrix_forward(avg_profile, gaps, beam_offsets)
        screen_avg = f_avg['screen']

        sp_screen_dict[n_bo].plot(screen_avg.x*1e3, screen_avg.intensity, label='%i' % (offset_error_s1*1e6))

    sp_screen.set_xlim(0, 3)
    sp_screen.set_ylim(0, 3e3)

    sp_profile.legend(title='Beam offset')
    sp_screen.legend(title='Beam offset')

sp_profile0.legend(title='Error [$\mu$m] / $\sigma$ [fs]')


for sp, xlim, ylim in zip(sp_screen_dict.values(), [2, 3, 2], [3e3, 1.5e3, 4e3]):
    sp.set_xlim(0, xlim)
    sp.set_ylim(0, ylim)
    sp.legend()

#ms.saveall('./group_metting_2020-11-17/offset_scan')

plt.show()

