import numpy as np
import pickle
import matplotlib.pyplot as plt

import myplotstyle as ms

plt.close('all')

with open('./image_obj.pkl', 'rb') as f:
    image_obj = pickle.load(f)

image = image_obj.image
x_axis = image_obj.x_axis
y_axis = image_obj.y_axis

n_slices = 50
new_img = image_obj.slice_x(n_slices)

ms.figure('Image analysis')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp = sp0 = subplot(sp_ctr, title='Original image', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1
image_obj.plot_img_and_proj(sp, x_factor=1e15)

sp_re = subplot(sp_ctr, title='Reshaped image', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1
new_img.plot_img_and_proj(sp_re, x_factor=1e15)

sp_ctr = np.inf
ny, nx = 3, 3
subplot = ms.subplot_factory(ny, nx)

slice_dict = new_img.fit_slice()

plot_gaussfits = False

if plot_gaussfits:
    for n_slice in range(n_slices):
        gf = slice_dict['slice_gf'][n_slice]

        if sp_ctr > ny*nx:
            ms.figure('Gaussfit details')
            sp_ctr = 1
        sp = subplot(sp_ctr, xlabel='y axis', ylabel='Slice intensity', title='Slice %i' % n_slice)
        sp_ctr += 1

        sp.plot(gf.xx, gf.yy)
        sp.plot(gf.xx, gf.reconstruction)

slice_mean = slice_dict['slice_mean']
slice_std = np.sqrt(slice_dict['slice_sigma'])
for sp_ in sp_re, sp0:
    sp_.errorbar(new_img.x_axis*1e15, slice_mean*1e3, yerr=slice_std*1e3, ls='None', marker='_', color='red')

plt.show()

