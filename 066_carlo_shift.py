import numpy as np
from socket import gethostname

import streaker_calibration
import elegant_matrix
import tracking
import config
import myplotstyle as ms

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-06-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/06/19/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-06-19/'


data_files = [
        (data_dir+ '2021_06_19-17_30_57_Calibration_SARUN18-UDCP020.h5', data_dir + '2021_06_19-17_50_16_Calibration_SARUN18-UDCP020.h5'),
        ]


for files in data_files:
    calibrator = sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3, fit_gap=True, fit_order=False)
    for _file in files:
        calibrator.add_file(_file)

    calibrator.fit()
    streaker_center = calibrator.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]['streaker_offset']
    gap_arr = np.array([10e-3-150e-6, 10e-3+50e-6])
    tracker_kwargs = config.get_default_tracker_settings()
    gauss_kwargs = config.get_default_gauss_recon_settings()
    gauss_kwargs['sig_t_range'] = np.arange(5, 65.01, 5)*1e-15
    tracker = tracking.Tracker(**tracker_kwargs)
    meta_data = calibrator.meta_data
    tracker.set_simulator(meta_data)

    gap_recon_dict = calibrator.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_center)


ms.show()

