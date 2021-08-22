import matplotlib.pyplot as plt
import numpy as np

import h5_storage
import streaker_calibration
import tracking
import config
import elegant_matrix

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

plt.close('all')

file_ = '/storage/data_2021-07-20/2021_07_20-12_23_34_Calibration_data_SATMA02-UDCP045.h5'
dict_ = h5_storage.loadH5Recursive(file_)
sc = streaker_calibration.StreakerCalibration('Athos', 0, 10e-3, 200e-12, file_)
sc.fit()
calib_dict = sc.fit_type('centroid')
sc.plot_streaker_calib()
streaker_offset = calib_dict['streaker_offset']



gap_arr = np.array([10e-3-100e-6, 10e-3+50e-6])
tracker_kwargs = config.get_default_tracker_settings()
tracker_kwargs['beamline'] = 'Athos'
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(dict_['meta_data_begin'])
print(tracker.calcR12())
gauss_kwargs = config.get_default_gauss_recon_settings()
gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_offset, gap0=10e-3)
sc.plot_gap_reconstruction(gap_recon_dict)


sc.reconstruct_current(tracker, gauss_kwargs)
sc.plot_reconstruction(max_distance=np.inf)

plt.show()

