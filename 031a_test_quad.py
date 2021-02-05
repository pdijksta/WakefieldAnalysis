import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname

import tracking
import elegant_matrix


import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

len_profile = int(5e3)
charge = 200e-12
energy_eV = 4.5e9
struct_lengths = [1., 1.]
n_particles = int(1e5)
n_emittances = [850e-9, 850e-9]
screen_bins = 500
screen_cutoff = 1e-3
smoothen = 25e-6
profile_cutoff = 0
timestamp = 1601761132
gaps = [10e-3, 10e-3]
beam_offsets = [0, 4.692e-3]
n_streaker = 1

hostname = gethostname()
if hostname == 'desktop':
    dirname1 = '/storage/data_2020-10-03/'
    dirname2 = '/storage/data_2020-10-04/'
    archiver_dir = '/storage/Philipp_data_folder/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2020/10/03/'
    dirname2 = '/sf/data/measurements/2020/10/04/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2020-10-03/'
    dirname2 = '/home/work/data_2020-10-04/'
    archiver_dir = '/home/work/'



bp_gauss = tracking.get_gaussian_profile(40e-15, 200e-15, len_profile, charge, energy_eV)

flat_current = np.zeros_like(bp_gauss.current)
flat_time = bp_gauss.time
flat_current[np.logical_and(flat_time > -40e-15, flat_time < 40e-15)] = 1

bp_flat = tracking.BeamProfile(flat_time, flat_current, energy_eV, charge)




for bp, bp_label in [(bp_gauss, 'Gauss'), (bp_flat, 'Flat')]:

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

        tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake)
        label = main_label + ' ' + bp_label
        forward_dict = tracker.matrix_forward(bp, gaps, beam_offsets)
        screen = forward_dict['screen']
        #screen.smoothen(15e-6)
        #screen.cutoff(1e-3)

        if not plot_wake:
            plot_wake = True
            wake_dict = forward_dict['wake_dict'][n_streaker]
            quad_wake = wake_dict['quad']*forward_dict['bs_at_streaker'][n_streaker]

            sp_wake.plot(wake_dict['wake_t'], wake_dict['wake'], label='Dipole %s' % bp_label)
            sp_wake.plot(wake_dict['wake_t'], quad_wake, label='Quadrupole %s' % bp_label)



        bp_back = tracker.track_backward2(screen, bp, gaps, beam_offsets, n_streaker)
        new_bp_back = tracking.BeamProfile(
                bp_gauss.time,
                np.interp(bp_gauss.time, bp_back.time, bp_back.current, left=0., right=0.),
                energy_eV, charge)

        new_bp_back.plot_standard(sp_profile, label=label+' Rec')
        color = screen.plot_standard(sp_screen, label=label)[0].get_color()

        forward_dict2 = tracker.matrix_forward(new_bp_back, gaps, beam_offsets, debug=True)
        screen2 = forward_dict2['screen']
        screen2.plot_standard(sp_screen, color=color, ls='--')



sp_screen.legend()
sp_profile.legend()
sp_wake.legend()


plt.show()


