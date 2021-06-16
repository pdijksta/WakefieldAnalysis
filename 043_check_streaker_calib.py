from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import h5_storage
import misc2 as misc
import analysis
import myplotstyle as ms

plt.close('all')

screen0 = 555e-6
streaker0 = 364e-6

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-03-16/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/03/16/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-03-16/'

fig, plot_handles = analysis.streaker_calibration_figure()
fig.subplots_adjust(wspace=0.3)


streaker_calib_file = data_dir + '2021_03_16-20_07_45_Calibration_SARUN18-UDCP020.h5'

streaker_calib = h5_storage.loadH5Recursive(streaker_calib_file)
#streaker_calib['raw_data']['streaker_offsets'] = streaker_calib['raw_data']['streaker_offsets'][:-1]
#streaker_calib['raw_data']['pyscan_result']['image'] = streaker_calib['raw_data']['pyscan_result']['image'][:-1]

analyzed = analysis.analyze_streaker_calibration(streaker_calib['raw_data'], plot_handles=plot_handles)
print('Streaker offset [um]: %.3f' % (analyzed['meta_data']['streaker_offset']*1e6))
images = streaker_calib['raw_data']['pyscan_result']['image'].astype(float)
x_axis = streaker_calib['raw_data']['pyscan_result']['x_axis'].astype(float)
streaker_offsets = streaker_calib['raw_data']['streaker_offsets'] - 377e-6

n_offsets = images.shape[0]
n_images = images.shape[1]

#sp_ctr = np.inf
#nx, ny = 3, 3
#subplot = ms.subplot_factory(ny, nx)
#
#
#for n_offset in range(n_offsets):
#    if sp_ctr > nx*ny:
#        ms.figure('Streaker calibration')
#        plt.subplots_adjust(hspace=0.4)
#        sp_ctr = 1
#
#    sp = subplot(sp_ctr, title='Offset=%.2f mm' % (streaker_offsets[n_offset]*1e3), scix=True, sciy=True, xlabel='x axis')
#    sp_ctr += 1
#
#    for n_image in range(images.shape[1]):
#        image = images[n_offset, n_image]
#        projx = image.sum(axis=-2)
#        sp.plot(x_axis-screen0, projx)

screen_cutoff = 5e-2

rms_arr = np.zeros([n_offsets, n_images])
for n_offset in range(n_offsets):
    for n_image in range(n_images):
        proj = np.sum(images[n_offset, n_image], axis=-2)
        screen = misc.proj_to_screen(proj, x_axis, True, 555e-6)
        screen.cutoff2(screen_cutoff)
        screen.crop()
        screen.reshape(int(1e3))
        rms_arr[n_offset, n_image] = screen.rms()




sp = plot_handles[0].twinx()
sp.set_ylabel('RMS beamsizes [mm]')

rms_median = np.median(rms_arr, axis=1)

def fit_func(xx, mean, strength):
    return (xx-mean)**2 * strength + rms_median[6]

p_opt, p_cov = curve_fit(fit_func, streaker_offsets, rms_median, p0=[0, 1])

xx_fit = np.linspace(streaker_offsets.min(), streaker_offsets.max(), 1000)
yy_rec = fit_func(xx_fit, *p_opt)

rms_err = np.array([rms_median - rms_arr.min(axis=1), rms_arr.max(axis=1)-rms_median])
sp.errorbar(streaker_offsets*1e3, rms_median*1e3, rms_err*1e3, label='Beamsizes', color='red', ls='None', marker='.')
sp.plot(xx_fit*1e3, yy_rec*1e3, color='red', ls='--')
ms.comb_legend(plot_handles[0], sp)




#screen_data_file = data_dir + '2021_03_16-20_22_26_Screen_data_SARBD02-DSCR050.h5'
#screen_data = h5_storage.loadH5Recursive(screen_data_file)
#
#images = screen_data['pyscan_result']['image'].astype(float)
#x_axis = screen_data['pyscan_result']['x_axis'].astype(float)
#sp_ctr = np.inf
#screen0 = 555e-6
#
#if sp_ctr > nx*ny:
#    ms.figure('Screen data')
#    plt.subplots_adjust(hspace=0.4)
#    sp_ctr = 1
#
#sp = subplot(sp_ctr, title='Offset=?', scix=True, sciy=True, xlabel='x axis')
#sp_ctr += 1
#sp.set_xlim(-2e-3, 0.5e-3)
#
#for n_image in range(images.shape[0]):
#    image = images[n_image]
#    projx = image.sum(axis=-2)
#    sp.plot(x_axis-screen0, projx)







plt.show()

