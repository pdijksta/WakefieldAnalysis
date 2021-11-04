import numpy as np
from socket import gethostname

import image_and_profile as iap
import elegant_matrix
import tracking
import config
import h5_storage
import myplotstyle as ms

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

hostname = gethostname()
if hostname == 'desktop':
    data_dir1 = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir1 = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir1 = '/mnt/data/data_2021-05-18/'

file_ = data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'
dict_ = h5_storage.loadH5Recursive(file_)

gap_calibration = -63e-6
streaker_calibration = 376e-6
screen_x0 = 2791e-6


n_d = 0
n_d0 = -1

offset = dict_['raw_data']['streaker_offsets'][n_d]
image = dict_['raw_data']['pyscan_result']['image'][n_d, 0]
proj = image.sum(axis=0)[::-1]
x_axis_m = dict_['raw_data']['pyscan_result']['x_axis_m'][::-1]

proj0 = dict_['raw_data']['pyscan_result']['image'][n_d0, 0].sum(axis=0)[::-1]
screen_x0 = np.sum(proj0*x_axis_m) / np.sum(proj0)

tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()


screen0 = iap.ScreenDistribution(x_axis_m-screen_x0, proj0, subtract_min=True)
screen = iap.ScreenDistribution(x_axis_m-screen_x0, proj, subtract_min=True)
screen.cutoff2(tracker_kwargs['screen_cutoff'])
screen.crop()
screen.reshape(tracker_kwargs['len_screen'])

ms.figure()
subplot = ms.subplot_factory(2,2)
sp_ctr = 1
sp_screen = subplot(sp_ctr, title='Screen')
sp_ctr += 1

screen.plot_standard(sp_screen)
screen0.plot_standard(sp_screen)





tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(dict_['raw_data']['meta_data_begin'])

recon_kwargs['meas_screen'] = screen
recon_kwargs['gaps'] = [0., 10e-3 + gap_calibration]
recon_kwargs['charge'] = 180e-12
recon_kwargs['n_streaker'] = 1
recon_kwargs['beam_offsets'] = [0., -(offset - streaker_calibration)]

profiles = []
rms_durations = []
for _ in range(20):
    gauss_dict = tracker.find_best_gauss2(**recon_kwargs)
    profile = gauss_dict['reconstructed_profile']
    profiles.append(profile)
    rms_duration = profile.rms()
    rms_durations.append(rms_duration)


sp_profile = subplot(sp_ctr, title='Profiles')
sp_ctr += 1
for profile in profiles:
    profile.center('Mean')
    profile.plot_standard(sp_profile)


rms_durations = np.array(rms_durations)




ms.show()

