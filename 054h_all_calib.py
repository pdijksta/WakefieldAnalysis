from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

ms.closeall()
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
gauss_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)

for streaker_calib_files in all_streaker_calib:
    sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3)
    for scf in streaker_calib_files:
        sc.add_file(scf)
    sc.fit()
    sc.forward_propagate(blmeas_file, gauss_kwargs['tt_halfrange'], 200e-12, tracker, blmeas_cutoff=5e-2)
    sc.plot_streaker_calib()
    ms.plt.suptitle(scf)

ms.show()

