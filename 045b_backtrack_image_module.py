import numpy as np; np
import pickle

import image_and_profile as iap
import myplotstyle as ms

ms.plt.close('all')

with open('./backtrack_image_no_compensate.pkl', 'rb') as f:
    d = pickle.load(f)
    image = d['image']
    x_axis = d['x_axis']
    y_axis = d['y_axis']
    final_profile = d['final_profile']
    xx = d['xx']
    tt = d['tt']
    meas_screen = d['meas_screen']

if xx[1] < xx[0]:
    xx = xx[::-1]
    tt = tt[::-1]

image_obj = iap.Image(image, x_axis, y_axis)
image_cut = image_obj.cut(xx.min(), xx.max())
image2 = image_cut.reshape_x(len(final_profile))

figure = ms.figure('Backtrack image')
ms.plt.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2,3, grid=False)
sp_ctr = 1


sp = subplot(sp_ctr, title='X space 1', xlabel='x [mm]', ylabel='y [mm]')
sp_ctr += 1
image_cut.plot_img_and_proj(sp)

sp = subplot(sp_ctr, title='X space 2', xlabel='x [mm]', ylabel='y [mm]')
sp_ctr += 1
image2.plot_img_and_proj(sp)

new_img = image2.x_to_t(xx, tt, debug=True, print_=False)
ms.plt.figure(figure.number)

sp = subplot(sp_ctr, title='T space', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1

new_img.plot_img_and_proj(sp, x_factor=1e15, revert_x=False)

forced_time = final_profile.time-final_profile.time.min()
forced_proj = final_profile.current
forced_img = new_img.force_projection(forced_time, forced_proj)

sp = subplot(sp_ctr, title='T space', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1
forced_img.plot_img_and_proj(sp, x_factor=1e15, revert_x=False)

sp = subplot(sp_ctr, title='Debug profile')
sp_ctr += 1

sp.plot(final_profile.time, final_profile.current/final_profile.current.max(), label='Profile')
yy = forced_img.image.sum(axis=-2)
sp.plot(forced_img.x_axis, yy/yy.max(), label='Forced img')


yy = new_img.image.sum(axis=-2)
sp.plot(new_img.x_axis, yy/yy.max(), label='New img')
#sp.plot(new_img.x_axis[::-1], yy/yy.max(), label='New img reverted')

# Manual backtracking of projection

proj = image2.image.sum(axis=-2)
x_axis = image2.x_axis

long_x_axis = np.linspace(x_axis[0], x_axis[-1], int(100e3))
long_proj = np.interp(long_x_axis, x_axis, proj)

t_interp0 = np.interp(long_x_axis, xx, tt)
intensity, bins = np.histogram(t_interp0, bins=100, weights=long_proj)
new_x_axis = (bins[1:] + bins[:-1])/2.

sp.plot(new_x_axis, intensity/intensity.max(), label='Manual backtrack')

sp.legend()

sp = subplot(sp_ctr, title='Debug projection')
sp_ctr += 1

sp.plot(meas_screen.x, meas_screen.intensity, label='Meas screen')
sp.plot(x_axis, proj, label='Image 2')

sp.legend()

import pickle
filename = './image_obj.pkl'
with open(filename, 'wb') as f:
    pickle.dump(forced_img, f)
print('Saved %s' % filename)

ms.plt.show()

