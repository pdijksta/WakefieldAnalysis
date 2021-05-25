import pickle
from socket import gethostname
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
import matplotlib.cm as cm

import h5_storage
import myplotstyle as ms

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    dirname1 = '/storage/data_2021-05-18/'
    dirname2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2021/05/18/'
    dirname2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2021-05-18/'
    dirname2 = '/home/work/data_2021-05-19/'



empty_snapshot = dirname2 + '20210519_121718_SARBD02-DSCR050_camera_snapshot.h5'
image_dict = h5_storage.loadH5Recursive(empty_snapshot)
x_axis0 = image_dict['camera1']['x_axis']
y_axis0 = image_dict['camera1']['y_axis']

lasing_snapshot = dirname1+'20210518_233946_SARBD02-DSCR050_camera_snapshot.h5'

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


mask_bg = bg > 150

min_index_x = np.argwhere(x_axis0 == x_axis[0]).squeeze()
max_index_x = np.argwhere(x_axis0 == x_axis[-1]).squeeze()

min_index_y = np.argwhere(y_axis0 == y_axis[0]).squeeze()
max_index_y = np.argwhere(y_axis0 == y_axis[-1]).squeeze()

mask_image = mask_bg[min_index_y:max_index_y+1,min_index_x:max_index_x+1]


image2 = image.copy()
image_recover = image.copy()
image2[mask_image] = np.array([0], dtype=np.uint16)-1


mask_dest = image == 0

image_recover += bg[min_index_y:max_index_y+1,min_index_x:max_index_x+1]
image_recover[mask_dest] = 0


cmap = cm.get_cmap('viridis', 12)
image_float = image.astype(np.float64)

image_floored = image_float - np.median(image_float)
image_floored[image_floored < 0] = 0
image_normed = image_floored/np.max(image_floored)

colors = cmap(image_normed)

colors[mask_dest] = np.array([1, 0, 0, 1])




image_recover[mask_dest] = np.array([0], dtype=np.uint16)-1


for im, ax_x, ax_y, title in [
        (colors, x_axis, y_axis, 'Lasing'),
        (bg, x_axis0, y_axis0, 'Background'),
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
    sp.set_title(title)

plt.show()

