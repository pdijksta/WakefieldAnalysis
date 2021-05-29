import os
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
import analysis
import h5_storage
import config
import tracking
import elegant_matrix
import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

nums = plt.get_fignums()
print('plt.fignums', nums)
for num in nums:
    plt.figure(num).clf()
plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')

all_streaker_calib = [
        ('2021_05_18-17_08_40_Calibration_data_SARUN18-UDCP020.h5', False),
        #'2021_05_18-21_58_48_Calibration_data_SARUN18-UDCP020.h5', Bad saved data
        ('2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5', False),
        #('2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', True), # Affected but ok
        #('2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', True), # Affected but ok
        #'2021_05_19-00_13_25_Calibration_SARUN18-UDCP020.h5', # Bad data
        #'2021_05_19-00_24_47_Calibration_SARUN18-UDCP020.h5', # Bad data maybe
        #('2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', True),
        #('2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5', True),
        ]

blmeas_file = data_dir+'119325494_bunch_length_meas.h5'

ms.figure('Compare streaker calibrations')
subplot = ms.subplot_factory(1,2)
sp_ctr = 1
sp_center = subplot(sp_ctr, title='Centroids', xlabel='Streaker offset [mm]', ylabel='Centroid deflection [mm]')
sp_ctr += 1

sp_one_sided = subplot(sp_ctr, title='One sided', xlabel='Streaker offset [mm]', ylabel='Centroid deflection [mm]')
sp_ctr += 1


def streaker_calibration_fit_func_os(offsets, streaker_offset, strength, order, const, semigap):
    return const + strength * np.abs(offsets-streaker_offset-semigap)**(-order)

for ctr, (streaker_calib_file, one_sided) in enumerate(all_streaker_calib):


    file_ = data_dir+streaker_calib_file
    if not os.path.isfile(file_):
        file_ = data_dir2+streaker_calib_file

    input_dict = h5_storage.loadH5Recursive(file_)
    if 'raw_data' in input_dict:
        input_dict = input_dict['raw_data']

    magnet_data = input_dict['meta_data_begin']
    tracker_kwargs = config.default_tracker_settings.copy()
    tracker_kwargs['magnet_file'] = magnet_data
    tracker = tracking.Tracker(**tracker_kwargs)

    calib_dict = analysis.analyze_streaker_calibration(input_dict, do_plot=True, fit_order=True, fit_gap=False, debug=True, forward_propagate_blmeas=True, tracker=tracker, blmeas=blmeas_file)
    plt.suptitle(streaker_calib_file)
    #calib_dict2 = analysis.analyze_streaker_calibration(input_dict, do_plot=False, fit_order=False, fit_gap=False, debug=True)


    meta_data = calib_dict['meta_data']
    print('%s %i %i' % (streaker_calib_file, meta_data['streaker_offset']*1e6, meta_data['fit_dict_rms']['streaker_offset']*1e6))
    offsets = meta_data['offsets']
    streaker_offset = meta_data['streaker_offset']
    #streaker_offset2 = calib_dict2['meta_data']['streaker_offset']
    screen_x0 = meta_data['screen_x0']
    centroid_mean, centroid_std = meta_data['centroid_mean'], meta_data['centroid_std']
    gap_fit = meta_data['gap_fit']
    order_fit = meta_data['order_fit']
    print('%i' % (gap_fit*1e6), order_fit)

    sp_center.errorbar((offsets-streaker_offset)*1e3, (centroid_mean-screen_x0)*1e3, yerr=centroid_std*1e3, label='%i' % ctr)
    print('%s done' % streaker_calib_file)

    mask_positive = offsets > 0
    mask_negative = offsets < 0
    for mask, label in [(mask_positive, 'Pos'), (mask_negative, 'Neg')]:
        if np.any(mask):
            offsets_os = offsets[mask]
            cent_os = centroid_mean[mask]
            cent_err_os = centroid_std[mask]
            xx_plot = gap_fit/2. - np.abs(offsets_os - 370e-6)
            yy_plot = np.abs(cent_os - screen_x0)
            ls='--' if label == 'Pos' else None
            sp_one_sided.errorbar(xx_plot*1e3, yy_plot*1e3, yerr=cent_err_os*1e3, label='%i %s %i' % (ctr, label, streaker_offset*1e6), ls=ls)

sp_center.legend()
sp_one_sided.legend()

plt.show()

