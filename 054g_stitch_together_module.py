import os
import matplotlib.pyplot as plt
from socket import gethostname
import tracking
import config
import h5_storage
import elegant_matrix
import streaker_calibration as sc

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

blmeas_file = data_dir+'119325494_bunch_length_meas.h5'
data_dir2 = data_dir.replace('18', '19')

files1 = [
        data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        ]

files2 = [
        data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5',
        data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',
        ]

files3 = [
        data_dir+'2021_05_19-00_13_25_Calibration_SARUN18-UDCP020.h5', # Bad data
        data_dir+'2021_05_19-00_24_47_Calibration_SARUN18-UDCP020.h5', # Bad data maybe
        ]

for files in files1, files2:
    tracker_kwargs = config.get_default_tracker_settings()
    dict1 = h5_storage.loadH5Recursive(files[0])
    dict2 = h5_storage.loadH5Recursive(files[1])
    dict1 = sc.analyze_streaker_calibration(dict1['raw_data'], False)
    dict2 = sc.analyze_streaker_calibration(dict2['raw_data'], False)
    magnet_data = dict1['raw_data']['meta_data_begin']
    tracker_kwargs['magnet_file'] = magnet_data
    tracker_kwargs['quad_wake'] = False

    tracker = tracking.Tracker(**tracker_kwargs)
    #tracker.wake2d = True
    #tracker.split_streaker = 5
    #tracker.set_simulator(magnet_data)

    meta_data = sc.analyze_streaker_calibration_stitch_together(dict1, dict2, do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None, forward_propagate_blmeas=True, tracker=tracker, blmeas=blmeas_file, beamline='Aramis', charge=200e-12, fit_gap=False, debug=False, tt_halfrange=200e-15, len_screen=5000, force_gap=10e-3-70e-6)['meta_data']

    plt.suptitle(os.path.basename(files[0]))

    print(os.path.basename(files[0]))
    print(meta_data['streaker_offset'])
    print(meta_data['order_fit'])
    print('Gap fit', meta_data['gap_fit'])

plt.show()


orders = [3, 2.7]
gap1 = [9.9723, 9.9918]
gap2 = [9.9959, 9.900]

