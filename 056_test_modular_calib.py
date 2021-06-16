from socket import gethostname

import tracking
import streaker_calibration
import image_and_profile as iap
import config
import elegant_matrix
import myplotstyle as ms

config.fontsize = 8

ms.set_fontsizes(config.fontsize)
ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')
hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')
blmeas_file = data_dir+'119325494_bunch_length_meas.h5'
tt_halfrange = config.get_default_gauss_recon_settings()['tt_halfrange']
charge = 200e-12
streaker_calib_file = data_dir+'2021_05_18-17_08_40_Calibration_data_SARUN18-UDCP020.h5'

streaker_calib_files = [
        data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        ]


calib2 = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3)
calib2.add_file(streaker_calib_files[0])
calib2.add_file(streaker_calib_files[1])

tracker_kwargs = config.get_default_tracker_settings()
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(calib2.meta_data)

beam_profile = iap.profile_from_blmeas(blmeas_file, tt_halfrange, charge, tracker.energy_eV, True, 1)
beam_profile.reshape(tracker.len_screen)
beam_profile.cutoff2(5e-2)
beam_profile.crop()
beam_profile.reshape(tracker.len_screen)


calib2.fit()

calib2.forward_propagate(beam_profile, tt_halfrange, charge, tracker, type_='centroid')

calib2.plot_streaker_calib()

ms.show()

