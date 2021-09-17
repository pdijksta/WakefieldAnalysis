import numpy as np
from socket import gethostname

import elegant_matrix
import tracking
import analysis
import config
import lasing
import h5_storage
import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

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
charge = 200e-12
pulse_energy = 210e-6

label_on_off = [
        ('20:00:00', '2021_06_19-20_01_21_Lasing_False_SARBD02-DSCR050.h5', '2021_06_19-20_02_41_Lasing_False_SARBD02-DSCR050.h5'),
        ('20:17:00', '2021_06_19-20_16_54_Lasing_True_SARBD02-DSCR050.h5', '2021_06_19-20_18_16_Lasing_False_SARBD02-DSCR050.h5'),
        ('20:25:00', '2021_06_19-20_25_36_Lasing_True_SARBD02-DSCR050.h5', '2021_06_19-20_26_46_Lasing_False_SARBD02-DSCR050.h5'),
        ('23:24:00 (12 pC)', '2021_06_19-23_23_38_Lasing_True_SARBD02-DSCR050.h5', '2021_06_19-23_25_35_Lasing_False_SARBD02-DSCR050.h5'),
        ('23:28:20 (6 pC)', '2021_06_19-23_28_52_Lasing_True_SARBD02-DSCR050.h5', '2021_06_19-23_29_51_Lasing_False_SARBD02-DSCR050.h5'),
        ('23:31:30 (17 pC)', '2021_06_19-23_32_03_Lasing_False_SARBD02-DSCR050.h5', '2021_06_19-23_33_02_Lasing_True_SARBD02-DSCR050.h5'),
        ('23:38:45 (17 pC + 2ps)', '2021_06_19-23_39_53_Lasing_True_SARBD02-DSCR050.h5', '2021_06_19-23_40_53_Lasing_False_SARBD02-DSCR050.h5'),
        ]

kwargs_recon = config.get_default_gauss_recon_settings()
kwargs_recon['delta_gap'] = np.array([0., delta_gap])
#kwargs_recon['sig_t_range'] = np.exp(np.linspace(np.log(25), np.log(250), 10))*1e-15
#kwargs_recon['tt_halfrange'] = 400e-15
streaker_centers = np.array([0., streaker_center])

tracker_kwargs = config.get_default_tracker_settings()
tracker = tracking.Tracker(**tracker_kwargs)

for label, lasing_on_file, lasing_off_file in label_on_off:
    lasing_on_file = data_dir+lasing_on_file
    lasing_off_file = data_dir+lasing_off_file


    for lasing_file in [lasing_off_file, lasing_on_file]:
        lasing_dict = h5_storage.loadH5Recursive(lasing_file)
        analysis.reconstruct_current(lasing_dict, n_streaker, beamline, tracker, rec_mode, kwargs_recon, screen_x0, streaker_centers, figsize=(20, 16))
        ms.plt.suptitle(label)

    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)

    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_center, delta_gap, tracker_kwargs, recon_kwargs=kwargs_recon, charge=charge, subtract_median=True, slice_factor=3)

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
            rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        rec_obj.process_data()
        las_rec_images[title] = rec_obj
        #rec_obj.plot_images('raw', title)
        #rec_obj.plot_images('tE', title)

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=0.5e3)
    las_rec.plot(figsize=(20, 16))
    ms.plt.suptitle(label)


ms.saveall('./plots/066c', ending='.pdf', empty_suptitle=False)

ms.show()