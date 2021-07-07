import numpy as np
from socket import gethostname
from scipy.constants import c

import h5_storage
import misc2 as misc
import image_and_profile as iap
import tracking
import elegant_matrix
import config
import gaussfit
import myplotstyle as ms

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')


hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')

gap = 10e-3
beam_offset = 4.7e-3
struct_length = 1

gauss_kwargs = config.get_default_gauss_recon_settings()
tracker_kwargs = config.get_default_tracker_settings()

n_emittance = 300e-9
tracker_kwargs['n_emittances'] = [n_emittance, n_emittance]

tracker = tracking.Tracker(**tracker_kwargs)


blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'
blmeas_profile = iap.profile_from_blmeas(blmeas_file, gauss_kwargs['tt_halfrange'], gauss_kwargs['charge'], 0, True)
blmeas_profile.cutoff2(0.03)
blmeas_profile.crop()
blmeas_profile.reshape(1000)

ms.figure('Resolution', figsize=(10, 8))
ms.plt.subplots_adjust(hspace=0.4, wspace=0.8)
subplot = ms.subplot_factory(2,3, grid=False)
sp_ctr = 1

image_file = data_dir1+'2021_05_18-21_02_13_Lasing_False_SARBD02-DSCR050.h5'
image_dict = h5_storage.loadH5Recursive(image_file)
meta_data1 = image_dict['meta_data_begin']

screen_calib_file = data_dir1+'2021_05_18-16_39_27_Screen_Calibration_SARBD02-DSCR050.h5'
screen_calib_dict = h5_storage.loadH5Recursive(screen_calib_file)

screen_calib_raw_image = screen_calib_dict['pyscan_result']['image'][0].astype(float)
x_axis_calib = screen_calib_dict['pyscan_result']['x_axis_m']
screen_x0 = gaussfit.GaussFit(x_axis_calib, screen_calib_raw_image.sum(axis=0)).mean
x_axis_calib -= screen_x0
y_axis_calib = screen_calib_dict['pyscan_result']['y_axis_m']
screen_calib_raw_image -= np.median(screen_calib_raw_image)
screen_calib_image = iap.Image(screen_calib_raw_image, x_axis_calib, y_axis_calib)

images = image_dict['pyscan_result']['image'].astype(float)
x_axis = image_dict['pyscan_result']['x_axis_m'] - screen_x0
y_axis = image_dict['pyscan_result']['y_axis_m']
projx = images.sum(axis=-2)
median_index = misc.get_median(projx, method='mean', output='index')
raw_image1 = images[median_index]
raw_image1 -= np.median(raw_image1)
image1 = iap.Image(raw_image1, x_axis, y_axis)


strong_streaking_file = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'
strong_streaking_dict = h5_storage.loadH5Recursive(strong_streaking_file)
meta_data2 = strong_streaking_dict['meta_data_begin']

strong_calib_file = data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'
strong_calib_dict = h5_storage.loadH5Recursive(strong_calib_file)
screen_x0 = strong_calib_dict['meta_data']['screen_x0']
index = np.argwhere(strong_calib_dict['meta_data']['offsets'] == 0)
raw_image = ((strong_calib_dict['raw_data']['pyscan_result']['image'])[index,0]).astype(float).squeeze()
raw_image2 = ((strong_calib_dict['raw_data']['pyscan_result']['image'])[0,0]).astype(float).squeeze()
x_axis = strong_calib_dict['raw_data']['pyscan_result']['x_axis_m'] - screen_x0
y_axis = strong_calib_dict['raw_data']['pyscan_result']['y_axis_m']
calib_image2 = iap.Image(raw_image, x_axis, y_axis)
image2 = iap.Image(raw_image2, x_axis, y_axis)



for strength_label, meta_data, image, calib_image in [
        ('Normal', meta_data1, image1, screen_calib_image),
        #('Strong', meta_data2, image2, calib_image2),
        ]:

    tracker.set_simulator(meta_data)
    blmeas_profile.energy_eV = tracker.energy_eV

    sp_profile = subplot(sp_ctr, title='Profile and wake', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    sp_wake = sp_profile.twinx()
    sp_wake.set_ylabel('Wake (kV/m)')


    blmeas_profile.plot_standard(sp_profile, color='black')

    wake_dict = blmeas_profile.calc_wake(gap, beam_offset, struct_length)

    wake_t = wake_dict['input']['charge_xx']/c + blmeas_profile.time.min()
    wake_E = wake_dict['dipole']['wake_potential']
    sp_wake.plot(wake_t*1e15, wake_E/1e3)

    sp_profile = subplot(sp_ctr, title='Resolution', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    sp_res = sp_profile.twinx()
    sp_res.set_ylabel('R (fs)')

    blmeas_profile.plot_standard(sp_profile, color='black')

    tracker.n_particles = int(200e3)
    res_dicts = []
    for quad_wake, label in [(False, 'Dipole %i nm' % (n_emittance*1e9)), (True, 'Quadrupole')]:
        tracker.quad_wake = quad_wake
        res_dict = iap.calc_resolution(blmeas_profile, gap, beam_offset, struct_length, tracker, 1)
        res = res_dict['resolution']
        res_t = res_dict['time']

        sp_res.plot(res_t*1e15, res*1e15, label=label)
        res_dicts.append(res_dict)

    #sp_image = subplot(sp_ctr, title='Raw image', xlabel='x (mm)', ylabel='y (mm)', grid=False)
    #sp_ctr += 1
    #image.plot_img_and_proj(sp_image)

    gfX = gaussfit.GaussFit(x_axis_calib, screen_calib_raw_image.sum(axis=0))
    beamsize = gfX.sigma

    #sp_screen_calib = subplot(sp_ctr, title='Screen calibration', xlabel='x (mm)', ylabel='y (mm)', grid=False)
    #sp_ctr += 1
    #calib_image.plot_img_and_proj(sp_screen_calib)

    resolution2 = beamsize / res_dicts[0]['streaking_strength']
    time2 = res_dicts[0]['time']

    sp_res.plot(time2*1e15, resolution2*1e15, label='Dipole measured beamsize')


    sp_res.set_ylim(0, 15)


sp_res.legend()


ms.show()

