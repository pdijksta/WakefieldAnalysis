import cProfile
import numpy as np
from socket import gethostname

import h5_storage
import config
import tracking
import analysis
import misc2 as misc
import elegant_matrix
import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

ms.closeall()

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir.replace('19', '18')


tracker_kwargs = config.get_default_tracker_settings()
tracker_kwargs['len_screen'] = 1000
tracker_kwargs['n_particles'] = int(50e3)
recon_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)

#data_file = data_dir+'2021_05_19-14_49_38_Lasing_False_SARBD02-DSCR050.h5'
#data_file = data_dir + '2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5'
data_file = data_dir1 + '2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5'
data_dict = h5_storage.loadH5Recursive(data_file)

raw_data = data_dict['raw_data']
meta_data = data_dict['meta_data']

offset_index = -1

gaps = [10e-3, 9.94e-3]
streaker_offset = 372e-6
beam_offsets = [0, -(meta_data['offsets'][offset_index]-streaker_offset)]

tracker.set_simulator(raw_data['meta_data_begin'])

projx = raw_data['pyscan_result']['image'][offset_index].astype(np.float64).sum(axis=-2)
x_axis = raw_data['pyscan_result']['x_axis_m']
median_proj = misc.get_median(projx)
meas_screen = misc.proj_to_screen(median_proj, x_axis, True, meta_data['screen_x0'])

recon_kwargs['meas_screen'] = meas_screen
recon_kwargs['gaps'] = gaps
recon_kwargs['beam_offsets'] = beam_offsets
recon_kwargs['n_streaker'] = 1
recon_kwargs['method'] = 'centroid'
recon_kwargs['sig_t_range'] = np.array([5., 75.])*1e-15

#analysis.current_profile_rec_gauss(tracker, recon_kwargs)

profile_output = cProfile.run('analysis.current_profile_rec_gauss(tracker, recon_kwargs)', filename='./profile.txt')



ms.show()

