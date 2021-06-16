from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
import myplotstyle as ms

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
tracker_kwargs['quad_wake'] = True
gauss_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)
n_streaker = 1

streaker_calib_files = (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',)
streaker_calib_files = (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5') # Affected but ok

sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3)
for scf in streaker_calib_files:
    sc.add_file(scf)

for fit_gap in [True, False]:
    sc.fit_gap = fit_gap
    #sc.fit_order = not fit_gap

    sc.fit()
    blmeas_profile = sc.forward_propagate(blmeas_file, gauss_kwargs['tt_halfrange'], 200e-12, tracker, blmeas_cutoff=5e-2)['blmeas_profile']
    sc.plot_streaker_calib()

    offset_list, gauss_dicts = sc.reconstruct_current(tracker, gauss_kwargs)
    sc.plot_reconstruction(blmeas_profile=blmeas_profile)

    gap = sc.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]['gap_fit']
    ms.plt.suptitle('Gaussian reconstruction; gap=%.3f mm' % (gap*1e3))

ms.show()

