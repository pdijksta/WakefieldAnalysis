import pickle
import numpy as np
import matplotlib.pyplot as plt

import h5_storage
import myplotstyle as ms

plt.close('all')

empty_snapshot = '/sf/data/measurements/2021/05/19/20210519_121718_SARBD02-DSCR050_camera_snapshot.h5'
image_dict = h5_storage.loadH5Recursive(empty_snapshot)
x_axis0 = image_dict['camera1']['x_axis']
y_axis0 = image_dict['camera1']['y_axis']

lasing_snapshot = '/sf/data/measurements/2021/05/18/20210518_233946_SARBD02-DSCR050_camera_snapshot.h5'

image_dict = h5_storage.loadH5Recursive(lasing_snapshot)

x_axis = image_dict['camera1']['x_axis']
y_axis = image_dict['camera1']['y_axis']
image = image_dict['camera1']['image']

ms.figure('Saved')
subplot = ms.subplot_factory(1,2, False)
sp_ctr = 1

with open('./bytes.pkl', 'rb') as f:
    bytes = pickle.load(f)
arr0 = np.frombuffer(bytes, dtype=np.uint16)
bg = arr0.reshape([2160, 2560])

for im, ax_x, ax_y in [
        (image, x_axis, y_axis),
        (bg, x_axis0, y_axis0),
        ]:
    sp = sp0 = subplot(sp_ctr)
    sp_ctr += 1

    #mask_x = np.logical_and(450 < ax_x, 1300 > ax_x)
    #mask_y = np.logical_and(2000 < ax_y, 3500 > ax_y)
    mask_x = np.ones_like(ax_x, dtype=bool)
    mask_y = np.ones_like(ax_y, dtype=bool)

    ax_x2 = ax_x[mask_x]
    ax_y2 = ax_y[mask_y]

    im2 = im[mask_y][:,mask_x]
    sp.imshow(im2, aspect='auto', extent=(ax_x2[0], ax_x2[-1], ax_y2[-1], ax_y2[0]))

plt.show()

