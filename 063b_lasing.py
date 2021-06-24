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


lasing_off_file = data_dir1+'2021_05_18-23_42_10_Lasing_True_SARBD02-DSCR050.h5'
lasing_on_file = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'

lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)

for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
    x_axis = data_dict['pyscan_result']['x_axis_m']
    y_axis = data_dict['pyscan_result']['y_axis_m']
    limits = 3.625e-3, 3.725e-3

    def correct_image(img, x_axis, y_axis, limits):
        image = iap.Image(img, x_axis, y_axis)



        index1 = np.argmin((image.x_axis-limits[0])**2)
        index2 = np.argmin((image.x_axis-limits[1])**2)+1
        new_image = image.image.copy()
        new_image[:,index1:index2-1] = new_image[:,index1][:, np.newaxis]
        proj = np.sum(new_image,axis=0)
        interp_proj = np.interp(image.x_axis[index1:index2], [image.x_axis[index1], image.x_axis[index2]], [proj[index1], proj[index2]])
        new_image[:,index1:index2] *= interp_proj/proj[index1:index2]
        image_corrected = iap.Image(new_image, image.x_axis, image.y_axis)

        return image, image_corrected


    old_pyscan_image = data_dict['pyscan_result']['image'].astype(float)
    new_pyscan_image = np.zeros_like(old_pyscan_image)

    for n_image, img in enumerate(old_pyscan_image):

        image, image_corrected = correct_image(img, x_axis, y_axis, limits)
        new_pyscan_image[n_image] = image_corrected.image

        if False:
            ms.figure('Repair data')
            subplot = ms.subplot_factory(2,2, grid=False)
            sp_ctr = 1

            sp_raw = subplot(sp_ctr, title='Image raw')
            sp_ctr += 1

            sp_corrected = subplot(sp_ctr, title='Image corrected')
            sp_ctr += 1
            image.plot_img_and_proj(sp_raw)
            image_corrected.plot_img_and_proj(sp_corrected)



    print('Warning! using only first 5 images')
    new_pyscan_dict = {
            'image': new_pyscan_image[:5],
            'x_axis_m': image_corrected.x_axis,
            'y_axis_m': image_corrected.y_axis,
            }

    data_dict['pyscan_result'] = new_pyscan_dict



n_streaker = 1
screen_x0 = 4250e-6
beamline = 'Aramis'
streaker_offset = 374e-6
gap = 10e-3 - 62e-6
tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()

las_rec_on = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, gap, tracker_kwargs)

las_rec_on.add_dict(lasing_on_dict)
las_rec_on.get_current_profiles(recon_kwargs, do_plot=False)
las_rec_on.set_median_profile()
las_rec_on.calc_wake()
las_rec_on.cut_images()
las_rec_on.convert_axes()




ms.show()

