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

#energy_eV = 6.14e9
gaps = [10e-3, 10e-3]
beam_offset_list = ([4.7e-3, 0], [4.75e-3, 0], [4.65e-3, 0])
offset_error = [0.02e-3, 0]
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.05
backtrack_cutoff = 0
len_profile = 1e3
struct_lengths = [1., 1.]
n_bins=300
smoothen = 0e-6
n_emittances = (300e-9, 300e-9)
n_particles = int(200e3)

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, n_particles=n_particles)
energy_eV = tracker.energy_eV


profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)

meas_screen_dict = {}

ms.figure('Different offsets (error=%.2f)' % offset_error[0])
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

profile_meas2 = copy.deepcopy(profile_meas)
profile_meas2.shift()
sp_profile.plot(profile_meas2.time*1e15, profile_meas2.current/profile_meas2.integral, label='Real', color='black', lw=3)

sp_screen = subplot(sp_ctr, title='Measured screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr += 1

f_dict0 = tracker.elegant_forward(profile_meas, gaps, [0., 0.])
meas_screen0 = f_dict0['screen']
meas_screen0.normalize()

for n_bo, beam_offsets in enumerate(beam_offset_list):
    beam_offsets2 = np.array(beam_offsets) + np.array(offset_error)
    f_dict = tracker.elegant_forward(profile_meas, gaps, beam_offsets2)
    meas_screen = f_dict['screen']
    meas_screen.normalize()
    meas_screen_dict[n_bo] = meas_screen

    label = '%.2f' % (beam_offsets[0]*1e3)
    color = sp_screen.plot(meas_screen.x*1e3, meas_screen.intensity, label=label)[0].get_color()

    sig_t_range = np.arange(20, 60.01, 5)*1e-15
    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets2, n_streaker, charge)
    sig_t_range2 = np.arange(gauss_dict['gauss_sigma']-3, gauss_dict['gauss_sigma']+3.01, 1)
    gauss_dict2 = tracker.find_best_gauss(sig_t_range2, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets2, n_streaker, charge)

    reconstructed_screen = gauss_dict2['reconstructed_screen']
    reconstructed_screen.normalize()


    reconstructed_profile = gauss_dict2['reconstructed_profile']
    bp = copy.deepcopy(reconstructed_profile)
    #bp.find_agreement(profile_meas)
    bp.shift()
    sp_profile.plot(bp.time*1e15, bp.current/bp.integral, label=label, color=color)

    final_baf = tracker.back_and_forward(meas_screen, meas_screen0, reconstructed_profile, gaps, beam_offsets2, n_streaker, output='Full')
    final_bp = final_baf['beam_profile']
    final_screen = final_baf['screen']
    final_screen.normalize()

    final_bp.shift()

    sp_profile.plot(final_bp.time*1e15, final_bp.current/final_bp.integral, ls='--', color=color)
    sp_screen.plot(final_screen.x*1e3, final_screen.intensity, ls='--', color=color)
    sp_screen.plot(reconstructed_screen.x*1e3, reconstructed_screen.intensity, ls='-.', color=color)

sp_profile.legend(title='Beam offset')
sp_screen.legend(title='Beam offset')

plt.show()

