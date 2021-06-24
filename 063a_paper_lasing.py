import numpy as np
from socket import gethostname

import h5_storage
import config
import analysis
import image_and_profile as iap
import myplotstyle as ms

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
blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'

blmeas_profile1 = iap.profile_from_blmeas(blmeas_file, 200e-15, 200e-12, 1, zero_crossing=1)
blmeas_profile2 = iap.profile_from_blmeas(blmeas_file, 200e-15, 200e-12, 1, zero_crossing=2)

for p in blmeas_profile1, blmeas_profile2:
    p.cutoff2(5e-2)
    p.crop()
    p.reshape(int(1e3))

n_streaker = 1
screen_x0 = 4250e-6

tracker_kwargs = config.get_default_tracker_settings()
kwargs_recon = config.get_default_gauss_recon_settings()

kwargs_recon['delta_gap'] = [0, -62e-6]
streaker_offsets = [0., 374e-6]


lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)


ms.figure('Comparison Lasing On / Off')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen_comp = subplot(sp_ctr, title='Measured projections', xlabel='x (mm)', ylabel='Intensity (arb. units)')
sp_ctr += 1
sp_profile_comp = subplot(sp_ctr, title='Reconstructed profiles', xlabel='x (mm)', ylabel='I (kA)')
sp_ctr += 1




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



    new_pyscan_dict = {
            'image': new_pyscan_image,
            'x_axis_m': image_corrected.x_axis,
            'y_axis_m': image_corrected.y_axis,
            }

    data_dict['pyscan_result'] = new_pyscan_dict

for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):

    output_dicts = analysis.reconstruct_current(data_dict, n_streaker, 'Aramis', tracker_kwargs, 'All', kwargs_recon, screen_x0, streaker_offsets, blmeas_file, do_plot=False)

    ms.figure('All current recon %s' % title)
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_screen = subplot(sp_ctr, title='Measured projections', xlabel='x (mm)', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_profile = subplot(sp_ctr, title='Reconstructed profiles', xlabel='x (mm)', ylabel='I (kA)')
    sp_ctr += 1

    rms_list = []
    for ctr, output_dict in enumerate(output_dicts):
        color = ms.colorprog(ctr, output_dicts)
        gauss_dict = output_dict['gauss_dict']
        gauss_dict['meas_screen'].plot_standard(sp_screen)
        profile = gauss_dict['reconstructed_profile']
        rms_list.append(profile.rms())
        profile.plot_standard(sp_profile, center='Mean', label='%.1f fs' % (rms_list[-1]*1e15))



    rms_index = np.argsort(rms_list)[len(rms_list)//2]

    gauss_dict = output_dicts[rms_index]['gauss_dict']

    gauss_dict['meas_screen'].plot_standard(sp_screen, color='red', lw=3)
    profile = gauss_dict['reconstructed_profile']
    label = '%.1f fs' % (rms_list[rms_index]*1e15)
    profile.plot_standard(sp_profile, center='Mean', color='red', lw=3, label=label)

    gauss_dict['meas_screen'].plot_standard(sp_screen_comp, lw=3)
    profile = gauss_dict['reconstructed_profile']
    label = '%s %.1f fs' % (title, rms_list[rms_index]*1e15)
    profile.plot_standard(sp_profile_comp, center='Mean', lw=3, label=label)


    for ctr, (p, ls) in enumerate([(blmeas_profile1, '--'), (blmeas_profile2, 'dotted')], 1):
        p.plot_standard(sp_profile, center='Mean', color='black', ls=ls, label='TDC %i %.1f' % (ctr, p.rms()*1e15))
        if main_ctr == 0:
            p.plot_standard(sp_profile_comp, center='Mean', color='black', ls=ls, label='TDC %i %.1f' % (ctr, p.rms()*1e15))






    sp_profile.legend()

sp_profile_comp.legend()


ms.show()

