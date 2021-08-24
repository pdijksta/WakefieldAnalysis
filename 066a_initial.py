import numpy as np
from socket import gethostname

import tracking
import analysis
import config
import h5_storage
import myplotstyle as ms

ms.closeall()

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-06-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/06/19/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-06-19/'


streaker_center = 389e-6
delta_gap = -42e-6
screen_x0 = 0.00099948
n_streaker = 1
beamline = 'Aramis'
rec_mode = 'Median'
kwargs_recon = config.get_default_gauss_recon_settings()
kwargs_recon['delta_gap'] = np.array([0., delta_gap])
streaker_centers = np.array([0., streaker_center])

tracker_kwargs = config.get_default_tracker_settings()

tracker = tracking.Tracker(**tracker_kwargs)

# Bad data

lasing_off_files = [
        data_dir+'2021_06_19-17_09_41_Lasing_False_SARBD02-DSCR050.h5',
        data_dir+'2021_06_19-17_11_51_Lasing_False_SARBD02-DSCR050.h5',
        ]

for lasing_off_file in lasing_off_files:
    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_off_dict['meta_data_begin']['SARUN18-UDCP020:GAP'] = 10
    lasing_off_dict['meta_data_end']['SARUN18-UDCP020:GAP'] = 10
    analysis.reconstruct_current(lasing_off_dict, n_streaker, beamline, tracker, rec_mode, kwargs_recon, screen_x0, streaker_centers)

ms.saveall('./plots/066a', ending='.pdf')

ms.show()
