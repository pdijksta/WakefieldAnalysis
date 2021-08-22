from socket import gethostname
import numpy as np

import elegant_matrix
import tracking
import config
import streaker_calibration
import myplotstyle as ms

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')



sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3, 200e-12)
for scf in (data_dir1+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'):
    sc.add_file(scf)


tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(sc.meta_data)

offset_dict = sc.fit_type('centroid')
streaker_offset = offset_dict['streaker_offset']
index = -1
meas_screen = sc.get_meas_screens()[index]
meas_screen.cutoff2(tracker.screen_cutoff)
meas_screen.crop()
meas_screen.reshape(tracker.len_screen)


gap_arr = np.array([10e-3-120e-6, 10e-3+20e-6])

delta_offsets = np.arange(50, 100.01, 10)*1e-6

use_offsets0 = [0, 1, 2, 3, 12, 13, 14, 15]

durations = []
durations_std = []
delta_gaps = []

gap_recon_dicts = []

for ctr, delta_offset in enumerate(delta_offsets):

    offsets = sc.offsets[sc.offsets != 0]
    distances = (10e-3-55e-6)/2. - np.abs(offsets - streaker_offset)
    delta_d_max = np.take(distances, use_offsets0).max()
    use_offsets = np.argwhere(np.logical_and(distances < delta_d_max, distances > delta_d_max - delta_offset)).squeeze()


    gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, recon_kwargs, streaker_offset, gap0=10e-3, use_offsets=use_offsets)
    print('assumed_bunch_duration %.2f' % (gap_recon_dict['beamsize']*1e15))
    print('assumed_bunch_uncertainty %.2f' % (gap_recon_dict['beamsize_rms']*1e15))
    delta_gap = gap_recon_dict['gap'] - 10e-3
    print('gap correction: %2.f um' % (delta_gap*1e6))
    sc.plot_gap_reconstruction(gap_recon_dict)
    ms.plt.suptitle('Case %i' % ctr)

    durations.append(gap_recon_dict['beamsize'])
    durations_std.append(gap_recon_dict['beamsize_rms'])
    delta_gaps.append(delta_gap)
    gap_recon_dicts.append(gap_recon_dict)

    #gap_arr = gap_recon_dict['gap_arr']
    #beamsize_arr = gap_recon_dict['all_rms'].mean(axis=1)
    #beamsize_plus = gap_recon_dict['beamsize'] + gap_recon_dict['beamsize_rms']
    #beamsize_minus = gap_recon_dict['beamsize'] - gap_recon_dict['beamsize_rms']
    #sort = np.argsort(gap_arr)
    #gap_plus = np.interp(beamsize_plus, beamsize_arr[sort], gap_arr[sort])
    #gap_minus = np.interp(beamsize_minus, beamsize_arr[sort], gap_arr[sort])
    #print('Gap plus / minus, %.2f, %.2f' % (gap_plus*1e6, gap_minus*1e6))



ms.show()

