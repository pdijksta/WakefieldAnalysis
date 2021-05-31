import matplotlib.pyplot as plt
from socket import gethostname
import analysis
import tracking
import config
import h5_storage

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')



files1 = [
        data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        ]

tracker_kwargs = config.get_default_tracker_settings()


dict1 = h5_storage.loadH5Recursive(files1[0])
magnet_data = dict1['raw_data']['meta_data_begin']
tracker_kwargs['magnet_file'] = magnet_data

tracker = tracking.Tracker(**tracker_kwargs)
#tracker.set_simulator(magnet_data)

meta_data = analysis.analyze_streaker_calibration_stitch_together(files1[0], files1[1], do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None, forward_propagate_blmeas=True, tracker=tracker, blmeas=None, beamline='Aramis', charge=200e-12, fit_gap=False, debug=False, tt_halfrange=200e-15, len_screen=5000)['meta_data']


plt.show()
