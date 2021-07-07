import sys
import numpy as np
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


# Full lasing, but saturation
lasing_on_fileF = data_dir1+'2021_05_18-23_42_10_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileF = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'

# Full lasing begin
lasing_off_fileFB = data_dir1+'2021_05_18-21_02_13_Lasing_False_SARBD02-DSCR050.h5'
lasing_on_fileFB = data_dir1+'2021_05_18-20_52_52_Lasing_True_SARBD02-DSCR050.h5'

# Short pulse begin
lasing_on_fileSB = data_dir1+'2021_05_18-21_08_24_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileSB = data_dir1+'2021_05_18-21_06_46_Lasing_False_SARBD02-DSCR050.h5'


# Short pulse
lasing_on_fileS = data_dir1+'2021_05_18-23_47_11_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileS = data_dir1+'2021_05_18-23_48_12_Lasing_False_SARBD02-DSCR050.h5'

#Two color pulse I=3 A, k=2
lasing_on_file2 = data_dir1+'2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_file2 = data_dir1+'2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'

screen_x00 = 4250e-6
screen_x02 = 898.02e-6

streaker_offset0 = 374e-6
streaker_offset2 = 364e-6

plot_repair = False


for lasing_on_file, lasing_off_file, pulse_energy, repair_data, screen_x0, streaker_offset in [
        #(lasing_on_fileF, lasing_off_fileF, 625e-6, True, screen_x00, streaker_offset0),
        #(lasing_on_fileS, lasing_off_fileS, 85e-6, True, screen_x00, streaker_offset0),
        (lasing_on_file2, lasing_off_file2, 180e-6, False, screen_x02, streaker_offset2),
        (lasing_on_fileFB, lasing_off_fileFB, 625e-6, False, screen_x02, streaker_offset2),
        (lasing_on_fileSB, lasing_off_fileSB, 85e-6, False, screen_x02, streaker_offset2),
        ]:

    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)


    if repair_data:

        for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
            x_axis = data_dict['pyscan_result']['x_axis_m']
            y_axis = data_dict['pyscan_result']['y_axis_m']
            limits = 3.65e-3, 3.75e-3

            limits_sat_x = 3.8e-3, 4.5e-3
            limits_sat_y = 3.e-3, 5e-3

            def correct_image(img, x_axis, y_axis, limits):
                image = iap.Image(img, x_axis, y_axis)

                index1 = np.argmin((image.x_axis-limits[0])**2)
                index2 = np.argmin((image.x_axis-limits[1])**2)+1
                new_image = image.image.copy()
                for index_y in range(new_image.shape[0]):
                    new_image[index_y,index1:index2] = np.interp(image.x_axis[index1:index2], [image.x_axis[index1], image.x_axis[index2]], [image.image[index_y,index1], image.image[index_y,index2]])
                image_corrected = iap.Image(new_image, image.x_axis, image.y_axis)

                return image, image_corrected

            def correct_saturation(img, x_axis, y_axis):
                where0 = np.logical_and(img==0, np.logical_and(x_axis > limits_sat_x[0], x_axis < limits_sat_x[1])[np.newaxis,:])
                where0 = np.logical_and(where0, np.logical_and(y_axis > limits_sat_y[0], y_axis < limits_sat_y[1])[:,np.newaxis])
                new_image = img.copy()
                for index_y in range(img.shape[0]):
                    if np.any(where0[index_y]):
                        for index_x in np.argwhere(where0[index_y]):
                            new_image[index_y, index_x] = new_image[index_y, index_x-1]

                return new_image

            old_pyscan_image = data_dict['pyscan_result']['image'].astype(float)
            new_pyscan_image = np.zeros_like(old_pyscan_image)

            for n_image, img in enumerate(old_pyscan_image):

                image, image_corrected = correct_image(img, x_axis, y_axis, limits)
                img_sat = correct_saturation(image_corrected.image, image_corrected.x_axis, image_corrected.y_axis)
                image_sat = iap.Image(img_sat, image_corrected.x_axis, image_corrected.y_axis)
                new_pyscan_image[n_image] = image_sat.image

                if plot_repair:
                    ms.figure('Repair data')
                    subplot = ms.subplot_factory(2,2, grid=False)
                    sp_ctr = 1

                    sp_raw = subplot(sp_ctr, title='Image raw')
                    sp_ctr += 1

                    sp_corrected = subplot(sp_ctr, title='Image corrected')
                    sp_ctr += 1

                    sp_sat = subplot(sp_ctr, title='Saturation')
                    sp_ctr += 1

                    image.plot_img_and_proj(sp_raw)
                    image_corrected.plot_img_and_proj(sp_corrected)
                    image_sat.plot_img_and_proj(sp_sat)

            new_pyscan_dict = {
                    'image': new_pyscan_image,
                    'x_axis_m': image_corrected.x_axis,
                    'y_axis_m': image_corrected.y_axis,
                    }

            data_dict['pyscan_result'] = new_pyscan_dict

    if plot_repair:
        ms.show()
        sys.exit()

    n_streaker = 1
    beamline = 'Aramis'
    delta_gap = -62e-6
    tracker_kwargs = config.get_default_tracker_settings()
    recon_kwargs = config.get_default_gauss_recon_settings()
    slice_factor = 3
    charge = 200e-12

    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True, slice_factor=slice_factor)
        #rec_obj.do_recon_plot = True

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
            rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        rec_obj.process_data()
        las_rec_images[title] = rec_obj
        #rec_obj.plot_images('raw', title)
        #rec_obj.plot_images('tE', title)

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.0e3)
    las_rec.plot()

ms.show()

