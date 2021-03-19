import numpy as np
import matplotlib.pyplot as plt

import h5_storage
import analysis
import myplotstyle as ms

plt.close('all')

screen0 = 555e-6
streaker0 = 364e-6

streaker_calib_file = '/home/work/data_2021-03-16/2021_03_16-20_07_45_Calibration_SARUN18-UDCP020.h5'

streaker_calib = h5_storage.loadH5Recursive(streaker_calib_file)
streaker_calib['raw_data']['streaker_offsets'] = streaker_calib['raw_data']['streaker_offsets'][:-1]
streaker_calib['raw_data']['pyscan_result']['image'] = streaker_calib['raw_data']['pyscan_result']['image'][:-1]

analyzed = analysis.analyze_streaker_calibration(streaker_calib['raw_data'])
print('Streaker offset [um]: %.3f' % (analyzed['meta_data']['streaker_offset']*1e6))
images = streaker_calib['raw_data']['pyscan_result']['image'].astype(float)
x_axis = streaker_calib['raw_data']['pyscan_result']['x_axis'].astype(float)
streaker_offsets = streaker_calib['raw_data']['streaker_offsets'] - 377e-6

n_offsets = images.shape[0]

sp_ctr = np.inf
nx, ny = 3, 3
subplot = ms.subplot_factory(ny, nx)


for n_offset in range(n_offsets):
    if sp_ctr > nx*ny:
        ms.figure('Streaker calibration')
        plt.subplots_adjust(hspace=0.4)
        sp_ctr = 1

    sp = subplot(sp_ctr, title='Offset=%.2f mm' % (streaker_offsets[n_offset]*1e3), scix=True, sciy=True, xlabel='x axis')
    sp_ctr += 1
    #if streaker_offset[n_offset] < 0:
    #    sp.set_xlim(-2e-3, 0.5e-3)

    for n_image in range(images.shape[1]):
        image = images[n_offset, n_image]
        projx = image.sum(axis=-2)
        sp.plot(x_axis-screen0, projx)


screen_data_file = '/home/work/data_2021-03-16/2021_03_16-20_22_26_Screen_data_SARBD02-DSCR050.h5'
screen_data = h5_storage.loadH5Recursive(screen_data_file)

images = screen_data['pyscan_result']['image'].astype(float)
x_axis = screen_data['pyscan_result']['x_axis'].astype(float)
sp_ctr = np.inf
screen0 = 555e-6

if sp_ctr > nx*ny:
    ms.figure('Screen data')
    plt.subplots_adjust(hspace=0.4)
    sp_ctr = 1

sp = subplot(sp_ctr, title='Offset=?', scix=True, sciy=True, xlabel='x axis')
sp_ctr += 1
sp.set_xlim(-2e-3, 0.5e-3)

for n_image in range(images.shape[0]):
    image = images[n_image]
    projx = image.sum(axis=-2)
    sp.plot(x_axis-screen0, projx)







plt.show()





