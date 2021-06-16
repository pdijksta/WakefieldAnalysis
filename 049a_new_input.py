import numpy as np
import matplotlib.pyplot as plt
import h5_storage
from socket import gethostname

import elegant_matrix
import image_and_profile as iap
import analysis
import tracking
import misc2 as misc
import myplotstyle as ms

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2020-10-03/'
    archiver_dir = '/storage/Philipp_data_folder/archiver_api_data/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2020/10/03/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-10-03/'

data_dir2 = data_dir.replace('03', '04')

magnet_file = archiver_dir + '2020-10-03.h5'
blmeas_files = [
        data_dir2 + 'Bunch_length_meas_2020-10-04_14-53-46.h5',
        data_dir2 + 'Bunch_length_meas_2020-10-04_14-55-20.h5',
        data_dir2 + 'Bunch_length_meas_2020-10-04_15-02-29.h5',
        ]


matfile = data_dir2 + 'Passive_data_20201004T172425.mat'

new_dict = h5_storage.loadH5Recursive(matfile.replace('.mat', '_new.h5'))

projx = new_dict['pyscan_result']['projx']
offsets = new_dict['streaker_offsets']
x_axis = new_dict['pyscan_result']['x_axis_m']


streaker_analysis = analysis.analyze_streaker_calibration(new_dict)

index0 = np.argwhere(offsets == 0).squeeze()
screen_center = streaker_analysis['meta_data']['centroid_mean'][index0]


print('Screen center', screen_center*1e6, 'um')
print('Streaker center', streaker_analysis['meta_data']['streaker_offset']*1e6, 'um')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

len_profile = int(5e3)
charge = -200e-12
struct_lengths = [1., 1.]
n_particles = int(1e5)
n_emittances = [500e-9, 500e-9]
screen_bins = 500
screen_cutoff = 1e-2
smoothen = 30e-6
profile_cutoff = 1e-2
gaps = [10e-3, 10e-3]
streaker_center = streaker_analysis['meta_data']['streaker_offset']
n_streaker = 1
tt_halfrange = 200e-15
bp_smoothen = 1e-15
invert_offset = True
quad_wake = False
override_quad_beamsize = False
timestamp = elegant_matrix.get_timestamp(2020, 10, 4, 17, 24, 25)



tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, bp_smoothen=bp_smoothen, quad_wake=quad_wake)


ms.figure('Tracking')
subplot = ms.subplot_factory(1, 3)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Current profile', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

for file_ctr, blmeas_file in enumerate(blmeas_files):
    for zero_crossing in (1, 2):
        blmeas = iap.profile_from_blmeas(blmeas_file, tt_halfrange, charge, tracker.energy_eV, True, zero_crossing)
        blmeas.reshape(len_profile)
        blmeas.cutoff2(profile_cutoff)
        blmeas.crop()
        blmeas.reshape(len_profile)
        blmeas.plot_standard(sp_profile, label='TDC %i zc %i %i fs' % (file_ctr, zero_crossing, blmeas.rms()*1e15), ls='--', center='Mean')

sp_screen_pos = subplot(sp_ctr, title='Screen distributions', xlabel='x (mm)', ylabel='intensity (arb. units)')
sp_ctr += 1


sp_screen_neg = subplot(sp_ctr, title='Screen distributions', xlabel='x (mm)', ylabel='intensity (arb. units)')
sp_ctr += 1


for n_offset, offset in enumerate(offsets):

    label = '%.2f mm' % (offset*1e3)
    projections = projx[n_offset]
    median_proj = misc.get_median(projections, method='std')

    screen = misc.proj_to_screen(median_proj, x_axis, True, x_offset=screen_center)
    screen.cutoff2(screen_cutoff)
    screen.crop()
    screen.reshape(len_profile)

    if offset == 0:
        for sp_screen in sp_screen_pos, sp_screen_neg:
            screen.plot_standard(sp_screen, label=label)
        continue
    elif offset > 0:
        sp_screen = sp_screen_pos
    elif offset < 0:
        sp_screen = sp_screen_neg


    color = screen.plot_standard(sp_screen, label=label)[0].get_color()

    beam_offsets = [0, -(offset - streaker_center)]
    forward_dict = tracker.elegant_forward(blmeas, gaps, beam_offsets)
    forward_screen = forward_dict['screen']
    forward_screen.plot_standard(sp_screen, color=color, ls='--')

sp_profile.legend()
sp_screen_pos.legend()
sp_screen_neg.legend()

ms.saveall('/tmp/049a_', ending='.pdf')


plt.show()

