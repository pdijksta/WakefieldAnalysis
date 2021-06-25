#import numpy as np
from socket import gethostname

import h5_storage
import elegant_matrix
import lasing
import config
import image_and_profile as iap
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


#Two color pulse I=3 A, k=2
lasing_on_file2 = data_dir1+'2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_file2 = data_dir1+'2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'

screen_x02 = 898.02e-6
streaker_offset2 = 364e-6

for lasing_on_file, lasing_off_file, pulse_energy, repair_data, screen_x0, streaker_offset in [
        (lasing_on_file2, lasing_off_file2, 180e-6, False, screen_x02, streaker_offset2),
        ]:
    n_streaker = 1
    beamline = 'Aramis'
    gap = 10e-3 - 62e-6
    tracker_kwargs = config.get_default_tracker_settings()
    recon_kwargs = config.get_default_gauss_recon_settings()
    n_slices = 50
    charge = 200e-12

    las_rec_images = {}

    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(n_slices, screen_x0, beamline, n_streaker, streaker_offset, gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True)

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
        rec_obj.process_data()
        las_rec_images[title] = rec_obj
        rec_obj.plot_images('raw', title)
        rec_obj.plot_images('tE', title, log=False)

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.5e3, key_mean='slice_mean_rms', key_sigma='slice_rms')
    las_rec.plot(plot_loss=False)
    ms.plt.suptitle('RMS')

    las_rec2 = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.5e3, key_mean='slice_mean', key_sigma='slice_sigma')
    las_rec2.plot(plot_loss=False)
    ms.plt.suptitle('GF')

    las_rec2 = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.5e3, key_mean='slice_cut_mean', key_sigma='slice_cut_rms')
    las_rec2.plot(plot_loss=False)
    ms.plt.suptitle('CUT')

slice_dict = las_rec_images['Lasing On'].slice_dicts[4]
iap.plot_slice_dict(slice_dict)

ms.show()

