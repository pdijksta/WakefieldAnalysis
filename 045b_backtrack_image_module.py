import numpy as np
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

new_img = image2.x_to_t(xx, tt, debug=True)
ms.plt.figure(figure.number)

sp = subplot(sp_ctr, title='T space', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1

new_img.plot_img_and_proj(sp, x_factor=1e15, revert_x=False)



forced_time = new_img.x_axis
forced_proj = np.interp(forced_time, final_profile.time, final_profile.current)
#forced_proj = forced_proj[::-1]

forced_img = new_img.force_projection(forced_proj)

sp = subplot(sp_ctr, title='T space', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1
forced_img.plot_img_and_proj(sp, x_factor=1e15, revert_x=False)

sp = subplot(sp_ctr, title='Debug')
sp_ctr += 1

sp.plot(forced_time, forced_proj/forced_proj.max(), label='Forced proj')
sp.plot(final_profile.time, final_profile.current/final_profile.current.max(), label='Profile')
yy = forced_img.image.sum(axis=-2)
sp.plot(forced_img.x_axis, yy/yy.max(), label='Forced img')


yy = new_img.image.sum(axis=-2)
sp.plot(new_img.x_axis, yy/yy.max(), label='New img')

sp.legend()

#import pickle
#filename = './image_obj.pkl'
#with open(filename, 'wb') as f:
#    pickle.dump(forced_img, f)
#print('Saved %s' % filename)

ms.plt.show()

