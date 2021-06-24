import argparse
import numpy as np
from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
#import analysis
import myplotstyle as ms

parser = argparse.ArgumentParser()
parser.add_argument('--file_index', type=int, default=2)
parser.add_argument('--method', type=str, default='centroid')
parser.add_argument('--noshow', action='store_true')
args = parser.parse_args()

file_index = args.file_index
method = args.method

ms.closeall()
elegant_matrix.set_tmp_dir('~/tmp_elegant/')
ms.set_fontsizes(config.fontsize)

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')

all_streaker_calib = [
        (data_dir+'2021_05_18-17_08_40_Calibration_data_SARUN18-UDCP020.h5',),
        #'2021_05_18-21_58_48_Calibration_data_SARUN18-UDCP020.h5', Bad saved data
        (data_dir+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5',),
        (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'), # Affected but ok
        #'2021_05_19-00_13_25_Calibration_SARUN18-UDCP020.h5', # Bad data
        #'2021_05_19-00_24_47_Calibration_SARUN18-UDCP020.h5', # Bad data maybe
        (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',),
        ]

blmeas_file = data_dir+'119325494_bunch_length_meas.h5'
tracker_kwargs = config.get_default_tracker_settings()
tracker_kwargs['quad_wake'] = False
tracker_kwargs['n_particles'] = int(50e3)
tracker_kwargs['profile_cutoff'] = 0.03
tracker_kwargs['bp_smoothen'] = 0
gauss_kwargs = config.get_default_gauss_recon_settings()
tt_halfrange = gauss_kwargs['tt_halfrange']
tracker = tracking.Tracker(**tracker_kwargs)
n_streaker = 1
#offset_index = -1

#streaker_calib_files = (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',)
#streaker_calib_files = (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5')
#streaker_calib_files = (data_dir+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5',)

streaker_calib_files = all_streaker_calib[file_index]

sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3, fit_gap=True, fit_order=False, proj_cutoff=tracker.screen_cutoff)
for scf in streaker_calib_files:
    sc.add_file(scf)
s_dict = sc.fit_type('centroid')
streaker_offset = s_dict['streaker_offset']
sc.fit_gap = False

streaker = config.streaker_names[sc.beamline][sc.n_streaker]
self = sc

gap_arr = np.arange(9.90, 10.0001, 0.01)*1e-3
tracker.set_simulator(self.meta_data)
gauss_kwargs['n_streaker'] = self.n_streaker
gauss_kwargs['method'] = method
gauss_kwargs['sig_t_range'] = np.exp(np.linspace(np.log(10), np.log(85), 15))*1e-15


gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_offset)

sc.plot_gap_reconstruction(gap_recon_dict, streaker_offset, figsize=(20,16))

ms.saveall('./album061/%s_file_%i' % (method, file_index), empty_suptitle=False)
if not args.noshow:
    ms.show()

