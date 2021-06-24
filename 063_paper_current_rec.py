import numpy as np
from socket import gethostname

import h5_storage
import streaker_calibration
import elegant_matrix
import config
import tracking
import image_and_profile as iap
import myplotstyle as ms

calibrate_gap = False

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant')

hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')

blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'


streaker_calib_files = (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',)
#streaker_calib_files = (data_dir1+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5')
#streaker_calib_files = (data_dir1+'2021_05_19-00_13_25_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_19-00_24_47_Calibration_SARUN18-UDCP020.h5')

strong_streak_file = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'

meta_data_strong = h5_storage.loadH5Recursive(strong_streak_file)['meta_data_begin']

n_streaker = 1
charge = 200e-12

tracker = tracking.Tracker(**config.get_default_tracker_settings())
sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3, fit_gap=True, fit_order=False, proj_cutoff=tracker.screen_cutoff)
for scf in streaker_calib_files:
    sc.add_file(scf)

streaker_offset = sc.fit_type('centroid')['streaker_offset']
tracker.set_simulator(sc.meta_data)
gauss_kwargs = config.get_default_gauss_recon_settings()

if calibrate_gap:
    gap_arr = np.array([10e-3-100e-6, 10e-3+50e-6])

    gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_offset)
    sc.plot_gap_reconstruction(gap_recon_dict, streaker_offset)

    gap = gap_recon_dict['gap']

    delta_gap = gap - 10e-3
    print('Gap (mm) %.3f' % (gap*1e3))
    print('Delta Gap (mm) %.3f' % (delta_gap*1e3))
else:
    delta_gap = -62e-6


sc.gap0 = 10e-3+delta_gap

#gauss_kwargs['delta_gap'] = [0, -62e-6]
gauss_kwargs['delta_gap'] = [0, 0]

blmeas_profile = iap.profile_from_blmeas(blmeas_file, gauss_kwargs['tt_halfrange'], charge, tracker.energy_eV)
blmeas_profile.cutoff2(5e-2)
blmeas_profile.crop()
blmeas_profile.reshape(tracker.len_screen)

sc.reconstruct_current(tracker, gauss_kwargs)
sc.plot_reconstruction(blmeas_profile=blmeas_profile)



ms.figure('Resolution')
subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Current profile and resolution', xlabel='t (fs)', ylabel='Current (kA)')
sp_ctr += 1

blmeas_profile.plot_standard(sp_profile, color='black')

sp_res = sp_profile.twinx()
sp_res.set_ylabel('R (fs)')
sp_ctr += 1


beam_offsets = -(sc.offsets - streaker_offset)
distances = sc.gap0/2. - np.abs(beam_offsets)
sort = np.argsort(distances)
tracker.quad_wake = True
tracker.n_particles = int(200e3)

n_res = 3



for n_meta, (meta_data, ls) in enumerate([(sc.meta_data, None), (meta_data_strong, '--')]):
    tracker.set_simulator(meta_data)
    for distance in [250e-6, 300e-6, 350e-6]:
        beam_offset = sc.gap0/2. - distance
        tt, res = iap.calc_resolution(blmeas_profile, sc.gap0, beam_offset, 1., tracker, n_streaker, bins=(75, 50))
        sp_res.plot(tt*1e15, res*1e15, label='Setting %i %i $\mu$m' % (n_meta, round(distance*1e6)), ls=ls)
    tracker.quad_wake = False

sp_res.set_ylim(0, 10)

sp_res.legend(title='Distance to jaw')











ms.show()

