import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from h5_storage import loadH5Recursive
import ImageSlicer.imageSlicer as slicer
import elegant_matrix

import myplotstyle as ms

transpose = True
n_slices = 31

plt.close('all')

data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
#data_dir = '/mnt/usb/work/data_2020-02-03/'
data_file = data_dir + 'Eloss_UNDbis.mat'
result_dict = loadH5Recursive(os.path.basename(data_file)+'_wake.h5')

timestamp = elegant_matrix.get_timestamp(2020, 2, 3, 21, 35, 8)

simulator = elegant_matrix.get_simulator('/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-02-03.json11')
_, disp_dict= simulator.get_elegant_matrix(1, timestamp)
energy_eV = simulator.get_data('SARBD01-MBND100:P-SET', timestamp) * 1e6


disp_factor = disp_dict['SARBD02.DSCR050'] / energy_eV

data_dict = loadmat(data_file)
x_axis = data_dict['x_axis'].squeeze() * 1e-6
y_axis = data_dict['y_axis'].squeeze() * 1e-6

n_images, n_gaps = data_dict['Image'].shape

if transpose:
    x_axis, y_axis = y_axis, x_axis

gap_list = result_dict['gap_list']

nx, ny = 3, 3
subplot = ms.subplot_factory(ny, nx)
subplot0 = ms.subplot_factory(2, 2)

fig0 = ms.figure('Energy change')
sp_ctr = 1

sp_current = subplot0(sp_ctr, title='Current')
sp_ctr += 1

sp_current2 = subplot0(sp_ctr, title='Current')
sp_ctr += 1

sp_ene = subplot0(sp_ctr, title='Energy change')
sp_ctr += 1

sp_ene2 = subplot0(sp_ctr, title='Energy change', xlabel='Position [mm]', ylabel='Energy change [MeV]')
sp_ctr += 1


all_y = np.zeros([len(gap_list), n_images, n_slices])
all_x = all_y.copy()
all_current = all_y.copy()

sp_ctr = np.inf
for n_gap, gap in enumerate(gap_list):
    fig1 = ms.figure('Details gap %.1f' % (gap*1e3))
    sp_ctr1 = 1

    for n_image in range(n_images):
        title = '%.1f mm %i' % (gap*1e3, n_gap)
        image = data_dict['Image'][n_image,n_gap].astype(np.float64)
        if not transpose:
            image = image.T
        slice_dict = slicer.image_analysis(image, x_axis, y_axis, 2, n_slices, title=title, do_plot=False)

        current = slice_dict['current_arr']
        mask_current = current > current.max()/4.

        yy = slice_dict['mean_sliceX'][mask_current]
        xx0 = slice_dict['mean_sliceY']
        xx = xx0[mask_current]

        if n_image % 5 == 0:
            sp_gap_details = subplot(sp_ctr1, scix=True, sciy=True)
            sp_ctr1 += 1
        sp_gap_details.plot(xx*1e-6, yy*1e-6)

        all_y[n_gap, n_image] = slice_dict['mean_sliceX']
        all_x[n_gap, n_image] = slice_dict['mean_sliceY']
        all_current[n_gap, n_image] = slice_dict['current_arr']

    interp_xx = np.linspace(all_x[n_gap, :, 0].max(), all_x[n_gap, :, -1].min(), n_slices)
    interp_yy = np.zeros([n_images, n_slices])
    interp_current = np.zeros([n_images, n_slices])
    for n_image in range(n_images):
        interp_yy[n_image] = np.interp(interp_xx, all_x[n_gap, n_image], all_y[n_gap, n_image])
        interp_current[n_image] = np.interp(interp_xx, all_x[n_gap, n_image], all_current[n_gap, n_image])

    sp_avg = subplot(sp_ctr1, scix=True, sciy=True, title='Avg')
    sp_ctr1 += 1
    sp_avg.errorbar(interp_xx, np.mean(interp_yy, axis=0), yerr=np.std(interp_yy, axis=0))

    sp_avg_current = subplot(sp_ctr1, scix=True, sciy=True, title='Avg current')
    sp_ctr1 += 1
    sp_avg_current.errorbar(interp_xx, np.mean(interp_current, axis=0), yerr=np.std(interp_current, axis=0))

    if sp_ctr > ny*nx:
        fig2 = ms.figure('View selected images')
        sp_ctr = 1
    plt.figure(fig2.number)

    sp = subplot(sp_ctr, title=title, grid=False)
    sp_ctr += 1
    sp.imshow(image.T, aspect='auto', extent=(y_axis[0], y_axis[-1], x_axis[-1], x_axis[0]))

    current = slice_dict['current_arr']
    mask_current = current > current.max()/4.

    yy = slice_dict['mean_sliceX'][mask_current]
    xx0 = slice_dict['mean_sliceY']
    xx = xx0[mask_current]

    sp_ene.plot(xx, yy, label=title)
    sp_current.plot(xx0, current, label=title)

    if n_gap % 3 == 0 or gap*1e3 <= 5:
        sp_ene2.errorbar(interp_xx*1e3, np.mean(interp_yy, axis=0)/disp_factor/1e6, yerr=np.std(interp_yy, axis=0)/disp_factor/1e6, label=title)
        sp_current2.errorbar(interp_xx, np.mean(interp_current, axis=0), yerr=np.std(interp_current, axis=0), label=title)

for sp_ in sp_ene, sp_current, sp_ene2, sp_current2:
    sp_.legend()

plt.show()

