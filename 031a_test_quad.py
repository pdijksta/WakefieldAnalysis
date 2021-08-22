import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname

import tracking
import h5_storage
import image_and_profile as iap
import elegant_matrix
import doublehornfit


import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

len_profile = int(5e3)
charge = 200e-12
energy_eV = 4.5e9
struct_lengths = [1., 1.]
n_particles = int(1e5)
n_emittances = [500e-9, 850e-9]
screen_bins = 500
screen_cutoff = 1e-3
smoothen = 30e-6
profile_cutoff = 0
timestamp = 1601761132
gaps = [10e-3, 10e-3]
beam_offsets = [0, 4.692e-3]
n_streaker = 1

hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')

blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'

sc_file = data_dir1+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
meta_data = h5_storage.loadH5Recursive(sc_file)['raw_data']['meta_data_begin']



bp_gauss = iap.get_gaussian_profile(40e-15, 200e-15, len_profile, charge, energy_eV)

flat_current = np.zeros_like(bp_gauss.current)
flat_time = bp_gauss.time
flat_current[np.logical_and(flat_time > -40e-15, flat_time < 40e-15)] = 1

bp_flat = tracking.BeamProfile(flat_time, flat_current, energy_eV, charge)

sig = 5e-15
dhf_current = doublehornfit.DoublehornFit.fit_func(flat_time, -20e-15, 20e-15, sig, sig, sig, sig, 0.5, 1, 1)
dhf_current *= charge / dhf_current.sum()

bp_dhf = tracking.BeamProfile(flat_time, dhf_current, energy_eV, charge)
bp_dhf.cutoff(1e-3)




for bp, bp_label in [(bp_gauss, 'Gauss'), (bp_flat, 'Flat'), (bp_dhf, 'Double horn')]:

    ms.figure('Test quadrupole wake %s emittance %i nm' % (bp_label, n_emittances[0]*1e9))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1


    sp_profile = subplot(sp_ctr, title='Profile')
    sp_ctr += 1

    sp_screen = subplot(sp_ctr, title='Screen distributions')
    sp_ctr += 1

    sp_wake = subplot(sp_ctr, title='Wakes')
    sp_ctr += 1

    bp.plot_standard(sp_profile, label='Gauss')

    plot_wake = False

    for main_label, quad_wake in [('Dipole', False), ('+Quad', True)]:
        #if quad_wake:
        #    continue

        tracker = tracking.Tracker(meta_data, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=True)
        label = main_label + ' ' + bp_label
        forward_dict = tracker.matrix_forward(bp, gaps, beam_offsets)
        screen = forward_dict['screen']
        #screen.smoothen(15e-6)
        #screen.cutoff(1e-3)

        tracker = tracking.Tracker(meta_data, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake)
        if not plot_wake:
            plot_wake = True
            wake_dict = forward_dict['wake_dict'][n_streaker]
            quad_wake = wake_dict['quad']*forward_dict['bs_at_streaker'][n_streaker]

            sp_wake.plot(wake_dict['wake_t'], wake_dict['wake'], label='Dipole %s' % bp_label)
            sp_wake.plot(wake_dict['wake_t'], quad_wake, label='Quadrupole %s' % bp_label)



        bp_back = tracker.track_backward2(screen, bp, gaps, beam_offsets, n_streaker)
        new_bp_back = tracking.BeamProfile(bp_gauss.time, np.interp(bp_gauss.time, bp_back.time, bp_back.current, left=0., right=0.), energy_eV, charge)

        new_bp_back.plot_standard(sp_profile, label=label+' Rec')
        color = screen.plot_standard(sp_screen, label=label)[0].get_color()

        forward_dict2 = tracker.matrix_forward(new_bp_back, gaps, beam_offsets)
        screen2 = forward_dict2['screen']
        screen2.plot_standard(sp_screen, color=color, ls='--')



sp_screen.legend()
sp_profile.legend()
sp_wake.legend()


plt.show()


