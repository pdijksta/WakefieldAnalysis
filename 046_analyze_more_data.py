import pickle
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np

from h5_storage import loadH5Recursive
import lasing
import image_and_profile as iap
import misc
import myplotstyle as ms

plt.close('all')

data_dir = '/mnt/data/data_2021-03-16/'
gap = 10e-3
struct_offset = 5.120e-3
struct_center = 377e-6
screen_center = 555e-6
dispersion = 0.4394238
r12 = 11.955637383155203
n_streaker = 1

with open('./045f_final_profile.pkl', 'rb') as f:
    final_profile = pickle.load(f)

energy_eV = final_profile.energy_eV
charge = 200e-12

gaps = [gap, gap]
beam_offsets = [0, -(struct_offset - struct_center)]
struct_lengths = [1, 1]

#Variation of corrector
data_files0 = [
    '/sf/data/measurements/2021/03/16/2021_03_16-20_34_31_Screen_data_SARBD02-DSCR050.h5',
    '/sf/data/measurements/2021/03/16/2021_03_16-20_35_12_Screen_data_SARBD02-DSCR050.h5',
    '/sf/data/measurements/2021/03/16/2021_03_16-20_37_58_Screen_data_SARBD02-DSCR050.h5',
    '/sf/data/measurements/2021/03/16/2021_03_16-20_41_05_Screen_data_SARBD02-DSCR050.h5',
    ]

wake_t, wake_x = final_profile.get_x_t(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker], r12)
n_slices = 50
len_profile = int(2e3)

data_files = [data_dir + os.path.basename(x) for x in data_files0]

#Lasing off
lasing_off_file0 = '/sf/data/measurements/2021/03/16/2021_03_16-20_42_57_Screen_data_SARBD02-DSCR050.h5'
lasing_off_file = data_dir + os.path.basename(lasing_off_file0)

median_indices = OrderedDict()
median_images = OrderedDict()

full_slice_dict = OrderedDict()

for ctr, (data_file, label) in enumerate([(lasing_off_file, 'Lasing Off'), (data_files[-1], 'Lasing On')]):
    data_dict = loadH5Recursive(data_file)
    images = data_dict['pyscan_result']['image'].astype(np.float64)
    x_axis = data_dict['pyscan_result']['x_axis'].astype(np.float64)
    y_axis = data_dict['pyscan_result']['y_axis'].astype(np.float64)

    projx = images.sum(axis=-2)

    median_index = misc.get_median(projx, output='index')
    median_indices[label] = median_index
    print(label, median_index)

    sp_ctr = np.inf
    ny, nx = 3, 3
    subplot = ms.subplot_factory(ny, nx)

    full_slice_dict[label] = np.zeros([len(images), 5, n_slices])

    if ctr == 1:
        ref_y = full_slice_dict['Lasing Off'][:,4,0].mean()

    for n_image, image in enumerate(images):
        image_obj = iap.Image(image, x_axis, y_axis, subtract_median=True, x_offset=screen_center)
        image_cut = image_obj.cut(wake_x.min(), wake_x.max())
        image_reshaped = image_cut.reshape_x(len_profile)
        image_t = image_reshaped.x_to_t(wake_x, wake_t, debug=False)
        if ctr == 0:
            ref_y = None
        image_tE, ref_y = image_t.y_to_eV(dispersion, energy_eV, ref_y)
        image_t_reduced = image_tE.slice_x(n_slices)
        slice_dict = image_t_reduced.fit_slice(charge=charge, smoothen_first=True, smoothen=1e6)
        full_slice_dict[label][n_image, 0] = slice_dict['slice_x']
        full_slice_dict[label][n_image, 1] = slice_dict['slice_mean']
        full_slice_dict[label][n_image, 2] = slice_dict['slice_sigma']
        full_slice_dict[label][n_image, 3] = slice_dict['slice_current']
        full_slice_dict[label][n_image, 4, 0] = ref_y


        if n_image == median_index:
            median_images[label] = image_obj

        if True:
            if sp_ctr > ny*nx:
                ms.figure('Images %s' % label)
                plt.subplots_adjust(hspace=0.35, wspace=0.3)
                sp_ctr = 1

            sp = subplot(sp_ctr, grid=False, title='Image %i' % n_image, xlabel='x [mm]', ylabel='y [mm]')
            sp_ctr += 1
            image_obj.plot_img_and_proj(sp)

filename = './full_slice_dict.pkl'
with open(filename, 'wb') as f:
    pickle.dump(full_slice_dict, f)
print('Saved %s' % filename)


image_on = median_images['Lasing On']
image_off = median_images['Lasing Off']
n_slices = 50
len_profile = int(2e3)
n_streaker=1

lasing_dict = lasing.obtain_lasing(image_off, image_on, n_slices, wake_x, wake_t, len_profile, dispersion, energy_eV, charge, debug=False)


ms.figure('Lasing analysis')
subplot = ms.subplot_factory(3,2)
sp_ctr = 1

image_dict = lasing_dict['all_images']
all_slice_dict = lasing_dict['all_slice_dict']

for label, subdict in image_dict.items():
    label2 = label.replace('_', ' ')

    #sp = subplot(sp_ctr, title=label2+' cut', xlabel='t [fs]', ylabel='$\Delta$E [MeV]')
    #sp_ctr += 1

    #image_cut = subdict['image_cut']
    #image_cut.plot_img_and_proj(sp)


    sp = subplot(sp_ctr, title=label2, xlabel='t [fs]', ylabel='$\Delta$E [MeV]', grid=False)
    sp_ctr += 1

    image_tE = subdict['image_tE']
    image_tE.plot_img_and_proj(sp)

    slice_dict = all_slice_dict[label]
    slice_x = slice_dict['slice_x']
    slice_mean = slice_dict['slice_mean']
    slice_sigma = slice_dict['slice_sigma']
    sp.errorbar(slice_x*1e15, slice_mean*1e-6, yerr=slice_sigma*1e-6, marker='_', color='red', ls='None')

sp = subplot(sp_ctr, title='Lasing', xlabel='t [fs]', ylabel='P [GW]')
sp_ctr += 1

slice_time = lasing_dict['slice_time']
power_Eloss = lasing_dict['power_Eloss']
power_Espread = lasing_dict['power_Espread']

sp.plot(slice_time, power_Eloss, color='red')
sp.plot(slice_time, power_Espread, color='blue')


plt.show()

