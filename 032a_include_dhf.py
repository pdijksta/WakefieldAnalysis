import os
import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname
from h5_storage import loadH5Recursive
import gaussfit
import tracking
import elegant_matrix
import misc

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

len_profile = int(5e3)
charge = 200e-12
energy_eV = 4.5e9
struct_lengths = [1., 1.]
n_particles = int(1e5)
n_emittances = [1000e-9, 500e-9]
screen_bins = 500
screen_cutoff = 1e-2
smoothen = 30e-6
profile_cutoff = 0
timestamp = 1601761132
gap2_correcting_summand = -22e-6
gaps = [10e-3, 10e-3+gap2_correcting_summand]
mean_struct2 = 472e-6 # see 026_script
beam_offsets38 = [0, 4.22e-3 + mean_struct2]
#beam_offsets25 = [0, 5.11e-3-mean_struct2]
n_streaker = 1
tt_halfrange = 200e-15
bp_smoothen = 1e-15



emittance_arr = np.array([1., 200., 500.,])*1e-9

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

magnet_file = archiver_dir + 'archiver_api_data/2020-10-03.h5'


blmeas38 = dirname1+'129833611_bunch_length_meas.h5'
#blmeas25 = dirname2 + 'Bunch_length_meas_2020-10-04_14-43-30.h5'
#blmeas25 = dirname2 + '129918532_bunch_length_meas.h5'
blmeas25 = dirname2 + 'Bunch_length_meas_2020-10-04_14-53-46.h5'



file0 = dirname1 + 'Passive_data_20201003T231958.mat'

file38 = dirname1 + 'Passive_data_20201003T233852.mat'
file25 = dirname2 + 'Passive_data_20201004T172425.mat'
file25b = dirname2 + 'Passive_data_20201004T163828.mat'


dict38 = loadH5Recursive(file38+'.h5')
dict25 = loadH5Recursive(file25+'.h5')
dict0 = loadH5Recursive(file0+'.h5')
dict25b = loadH5Recursive(file25b+'.h5')





subtract_min = True

def get_screen_from_proj(projX, x_axis, invert_x):
    if invert_x:
        xx, yy = (-x_axis[::-1]).copy(), (projX[::-1]).copy()
    else:
        xx, yy = x_axis.copy(), projX.copy()
    if subtract_min:
        yy -= yy.min()

    screen = tracking.ScreenDistribution(xx, yy)
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen

process_dict = {
        'Long': {
            'filename': file38,
            'main_dict': dict38,
            'proj0': dict0['projx'][-1],
            'x_axis0': dict0['x_axis']*1e-6,
            'n_offset': None,
            'filename0': file0,
            'blmeas': blmeas38,
            'flipx': False,
            'beam_offsets': beam_offsets38,
        },
        'Medium': {
            'filename': file25,
            'main_dict': dict25,
            'proj0': dict25['projx'][7],
            'x_axis0': dict25['x_axis']*1e-6,
            'n_offset': 0,
            'filename0': file25,
            'blmeas': blmeas25, ###
            'flipx': True,
            'beam_offsets': beam_offsets38,
            },
        }

for main_label, p_dict in process_dict.items():
    #if main_label != 'Medium':
    #    continue

    timestamp = misc.get_timestamp(os.path.basename(p_dict['filename0']))

    projx0 = p_dict['proj0']
    x_axis0 = p_dict['x_axis0']

    dict_ = p_dict['main_dict']
    file_ = p_dict['filename']
    x_axis = dict_['x_axis']*1e-6
    y_axis = dict_['y_axis']*1e-6
    n_offset = p_dict['n_offset']
    beam_offsets = p_dict['beam_offsets']

    if n_offset is None:
        projections = dict_['projx']
    else:
        projections = dict_['projx'][n_offset]

    if np.diff(x_axis0)[0] < 0:
        x_axis0 = x_axis0[::-1]
        invert_x0 = True

    if np.diff(x_axis)[0] < 0:
        x_axis = x_axis[::-1]
        invert_x = True
    else:
        invert_x = False

    all_mean = []
    for proj in projx0:
        screen = get_screen_from_proj(proj, x_axis, invert_x0)
        xx, yy = screen._xx, screen._yy
        gf = gaussfit.GaussFit(xx, yy)
        all_mean.append(gf.mean)

    mean0 = np.mean(all_mean)


    profile_meas = tracking.profile_from_blmeas(p_dict['blmeas'], tt_halfrange, charge, energy_eV)
    profile_dhf = tracking.dhf_profile(profile_meas)
    profile_dhf.cutoff(screen_cutoff)
    profile_dhf.crop()
    #profile_meas.flipx()

    for n_proj in range(3):

        screen0 = get_screen_from_proj(projections[n_proj], x_axis, invert_x0)
        screen0._xx = screen0._xx - mean0
        screen0.cutoff(3e-2)
        screen0.crop()



        ms.figure('Fit quad effect %s' % main_label)
        subplot = ms.subplot_factory(1,2)
        sp_ctr = 1

        sp_profile = subplot(sp_ctr, title='Current profile', xlabel='time [fs]', ylabel='I (arb. units)')
        sp_ctr += 1
        profile_meas.plot_standard(sp_profile, label='$\sigma$ %i fs' % (profile_meas.gaussfit.sigma*1e15))

        sp_profile.legend()

        sp_forward = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Intensity (arb. units)')
        sp_ctr += 1

        screen0.plot_standard(sp_forward, label='Measured', lw=3, color='black')

        ms.figure('Fit quad effect fitted %s' % main_label)
        subplot = ms.subplot_factory(1,2)
        sp_ctr = 1

        sp_profile2 = subplot(sp_ctr, title='Current profile', xlabel='time [fs]', ylabel='I (arb. units)')
        sp_ctr += 1
        profile_meas.plot_standard(sp_profile2)
        profile_dhf.plot_standard(sp_profile2)


        sp_forward2 = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Intensity (arb. units)')
        sp_ctr += 1

        screen0.plot_standard(sp_forward2, label='Measured', lw=3, color='black')


        tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, bp_smoothen=bp_smoothen)


        emittance = 1000e-9
        delta_offset_arr = np.array([0, 50, 80])*1e-6
        legend_title = '$\Delta offset'
        #for emittance in emittance_arr:
        for delta_offset in delta_offset_arr:
            label = '%i nm' % (emittance*1e9)
            label = '%i um' % (delta_offset*1e6)
            beam_offsets2 = [beam_offsets[0], beam_offsets[1] + delta_offset]
            tracker.n_emittances = [emittance, n_emittances[1]]
            forward_dict = tracker.matrix_forward(profile_meas, gaps, beam_offsets2)
            forward_dict2 = tracker.matrix_forward(profile_dhf, gaps, beam_offsets2)
            screen = forward_dict['screen_no_smoothen']
            screen2 = forward_dict2['screen_no_smoothen']
            screen.smoothen(smoothen)
            screen2.smoothen(smoothen)

            bs = forward_dict['bs_at_streaker'][1]

            screen.crop()
            screen2.crop()
            screen.plot_standard(sp_forward, label=label)
            color = screen.plot_standard(sp_forward2, label=label)[0].get_color()
            screen2.plot_standard(sp_forward2, color=color, ls='--')


        sp_forward.legend(title=legend_title)
        sp_forward2.legend(title=legend_title)

plt.show()

