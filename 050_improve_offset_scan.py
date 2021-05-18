import matplotlib.pyplot as plt
import socket

import h5_storage
import analysis

plt.close('all')

hostname = socket.gethostname()
if 'psi' in hostname or 'lc6a' in hostname or 'lc7a' in hostname:
    default_dir = '/sf/data/measurements/2021/04/25/'
    archiver_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/'
elif hostname == 'desktop':
    default_dir = '/storage/data_2021-04-25/'
    archiver_dir = '/storage/Philipp_data_folder/archiver_api_data/'
elif hostname == 'pubuntu':
    default_dir = '/home/work/data_2021-04-25/'
    archiver_dir = '/home/work/archiver_api_data/'

streaker_calib_file = default_dir + '2021_04_25-16_55_25_Calibration_SARUN18-UDCP020.h5'
data_dict = h5_storage.loadH5Recursive(streaker_calib_file)
analysis.analyze_streaker_calibration(data_dict['raw_data'])



plt.show()
