import os
import sys; sys
#import numpy as np
from socket import gethostname

import h5_storage
import elegant_matrix
import lasing
import config
import tracking
#import image_and_profile as iap
import streaker_calibration
import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

ms.closeall()

hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')


all_files = [
        #CENTER=5.12
        (5.12, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_19_53_Lasing_True_SARBD02-DSCR050.h5'),
        (5.12, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_22_06_Lasing_True_SARBD02-DSCR050.h5'),
        (5.12, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_31_24_Lasing_False_SARBD02-DSCR050.h5'),
        (5.12, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_33_02_Lasing_False_SARBD02-DSCR050.h5'),
        (5.12, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_42_37_Lasing_True_SARBD02-DSCR050.h5'),
        (5.12, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_43_47_Lasing_False_SARBD02-DSCR050.h5'),
        #CENTER=5.05'
        (5.05, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_23_21_Lasing_True_SARBD02-DSCR050.h5'),
        (5.05, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_24_37_Lasing_True_SARBD02-DSCR050.h5'),
        (5.05, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_30_09_Lasing_False_SARBD02-DSCR050.h5'),
        (5.05, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_35_21_Lasing_False_SARBD02-DSCR050.h5'),
        (5.05, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_40_03_Lasing_True_SARBD02-DSCR050.h5'),
        (5.05, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_46_23_Lasing_False_SARBD02-DSCR050.h5'),
        #CENTER=5.08'
        (5.08, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_25_51_Lasing_True_SARBD02-DSCR050.h5'),
        (5.08, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_27_16_Lasing_True_SARBD02-DSCR050.h5'),
        (5.08, 3, 0.5, '/sf/data/measurements/2021/05/18/2021_05_18-21_28_53_Lasing_False_SARBD02-DSCR050.h5'),
        (5.08, 3, 1, '/sf/data/measurements/2021/05/18/2021_05_18-21_34_25_Lasing_False_SARBD02-DSCR050.h5'),
        (5.08, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'),
        (5.08, 3, 2, '/sf/data/measurements/2021/05/18/2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'),
        ]

lasing_on_files = [x[-1] for x in all_files if 'Lasing_True' in x[-1]]
lasing_off_files = []

for file1 in lasing_on_files:
    for _a1, _a2, _a3, file2 in all_files:
        if file2 == file1:
            a1, a2, a3 = _a1, _a2, _a3
            break
    for _a1, _a2, _a3, file2 in all_files:
        if (_a1, _a2, _a3) == (a1, a2, a3) and file1 != file2:
            lasing_off_files.append(file2)
            break

streaker_calib_file = data_dir1+'/2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
streaker_calib = h5_storage.loadH5Recursive(streaker_calib_file)['raw_data']
tracker_kwargs = config.get_default_tracker_settings()
gauss_kwargs = config.get_default_gauss_recon_settings()
charge = 200e-12


if False:
    tracker = tracking.Tracker(**tracker_kwargs)
    tracker.set_simulator(streaker_calib['meta_data_begin'])
    calib_dict = streaker_calibration.reconstruct_gap(streaker_calib, tracker, gauss_kwargs, charge)
    streaker_offset = calib_dict['streaker_offset']
    screen_x0_arr = calib_dict['screen_x0']
    screen_x0 = screen_x0_arr[0]
    delta_gap = calib_dict['delta_gap']
else:
    delta_gap = 9.9929e-3 - 10e-3
    screen_x0 = 898e-6
    streaker_offset = 363.7e-6

n_streaker = 1
beamline = 'Aramis'
n_slices = 50

pulse_energy = 85e-6



for lasing_on_file, lasing_off_file in zip(lasing_on_files, lasing_off_files):

    lasing_on_file = data_dir1+os.path.basename(lasing_on_file)
    lasing_off_file = data_dir1+os.path.basename(lasing_off_file)
    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)


    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(n_slices, screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=gauss_kwargs, charge=charge, subtract_median=True)

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
        rec_obj.process_data()
        las_rec_images[title] = rec_obj
        #rec_obj.plot_images('raw', title)
        rec_obj.plot_images('tE', title)
        ms.saveall('./album063d/%s_imagestE' % os.path.basename(lasing_on_file), empty_suptitle=False)
        ms.closeall()

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.0e3)
    las_rec.plot()
    ms.saveall('./album063d/%s_lasing' % os.path.basename(lasing_on_file), empty_suptitle=False)
    ms.closeall()

ms.show()

