import numpy as np
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

order0_range = np.arange(2.6, 2.901, 0.05)
rec_gaps_rms = []
rec_gaps_centroid = []
rec_center_rms = []
rec_center_centroid = []

for order0 in order0_range:
    calib2.order_rms = order0
    calib2.order_centroid = order0

    calib2.fit()
    #calib2.forward_propagate(beam_profile, tt_halfrange, charge, tracker, type_='centroid')
    #fig, plot_handles = streaker_calibration.streaker_calibration_figure()
    #ms.plt.suptitle('Order=%.2f' % order0)
    #calib2.plot_streaker_calib(plot_handles)

    result_dict = calib2.get_result_dict()
    fit_dict_rms = result_dict['meta_data']['fit_dict_rms']
    rec_gaps_rms.append(fit_dict_rms['gap_fit']-10e-3)
    rec_center_rms.append(fit_dict_rms['streaker_offset'])

    fit_dict_centroid = result_dict['meta_data']['fit_dict_centroid']
    rec_gaps_centroid.append(fit_dict_centroid['gap_fit']-10e-3)
    rec_center_centroid.append(fit_dict_centroid['streaker_offset'])

rec_gaps_rms = np.array(rec_gaps_rms)
rec_gaps_centroid = np.array(rec_gaps_centroid)
rec_center_rms = np.array(rec_center_rms)
rec_center_centroid = np.array(rec_center_centroid)


ms.figure('Dependency of reconstructed gap on order')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1
sp_center = subplot(sp_ctr, xlabel='Fit order', ylabel='Streaker center ($\mu$m)', title='Reconstructed center')
sp_ctr += 1
sp_gap = subplot(sp_ctr, xlabel='Fit order', ylabel='$\Delta$ Streaker gap ($\mu$m)', title='Reconstructed gap correction')
sp_ctr += 1

sp_center.plot(order0_range, rec_center_centroid*1e6, label='Centroid', marker='.')
sp_center.plot(order0_range, rec_center_rms*1e6, label='Beamsize', marker='.')
sp_gap.plot(order0_range, rec_gaps_centroid*1e6, label='Centroid', marker='.')
sp_gap.plot(order0_range, rec_gaps_rms*1e6, label='Beamsize', marker='.')


sp_center.legend()
sp_gap.legend()

ms.show()

