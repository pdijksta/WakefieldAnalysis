import h5py
import os
import glob
from socket import gethostname
import numpy as np
import pickle
import matplotlib.cm as cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import myplotstyle as ms
import h5_storage

ms.mute = True

#plt.close('all')

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





with open('./bytes.pkl', 'rb') as f:
    bytes = pickle.load(f)
arr0 = np.frombuffer(bytes, dtype=np.uint16)
bg = arr0.reshape([2160, 2560])

empty_snapshot = dirname2 + '20210519_121718_SARBD02-DSCR050_camera_snapshot.h5'
image_dict = h5_storage.loadH5Recursive(empty_snapshot)
x_axis0 = image_dict['camera1']['x_axis']
y_axis0 = image_dict['camera1']['y_axis']


lasing_files = glob.glob(dirname1+'2021_*Lasing_*SARBD02*.h5')
lasing_files += glob.glob(dirname1+'*Calibration*.h5')

ny, nx = 4, 4
subplot = ms.subplot_factory(ny, nx, grid=False)

cmap = cm.get_cmap('viridis', 12)
lasing_snapshot = dirname1+'20210518_233946_SARBD02-DSCR050_camera_snapshot.h5'

error_files = []

def save_lasing(lasing_file):
    if 'Lasing_reconstruction' in lasing_file:
        return

    if os.path.basename(lasing_file) == '2021_05_18-18_11_27_Lasing_True_SARBD02-DSCR050.h5':
        return
    try:
        #data_dict = h5_storage.loadH5Recursive(lasing_file)
        with h5py.File(lasing_file, 'r') as data_dict:
            for key in 'raw_data', 'pyscan_result':
                if key in data_dict:
                    data_dict = data_dict[key]

            #data_dict = h5_storage.loadH5Recursive(lasing_snapshot)['camera1']
            if 'Calibration_SARUN' in lasing_file or 'Calibration_data_SARUN' in lasing_file:
                x_axis = np.array(data_dict['x_axis']).squeeze()[0,0]
                y_axis = np.array(data_dict['y_axis']).squeeze()[0,0]
                images0 = np.array(data_dict['image']).squeeze()
                shape = images0.shape
                images = images0.reshape([shape[0]*shape[1], shape[2], shape[3]])
            else:
                x_axis = np.array(data_dict['x_axis']).squeeze()[0]
                y_axis = np.array(data_dict['y_axis']).squeeze()[0]
                images = np.array(data_dict['image']).squeeze()
    except KeyError:
        print('Error for file %s' % os.path.basename(lasing_file))
        error_files.append(lasing_file)
        return

    extent = (x_axis[0], x_axis[-1], y_axis[-1], y_axis[0])

    sp_ctr = np.inf
    image_ctr = 1
    figs = []

    #if True:
    #    image = images
    #    n_image = 0
    for n_image, image in enumerate(images):

        if sp_ctr+1 > ny*nx:
            fig = Figure(figsize=(20, 16))
            plt.suptitle(os.path.basename(lasing_file)+' %i' % image_ctr)
            fig.subplots_adjust(hspace=0.4)
            image_ctr += 1
            sp_ctr = 1
            figs.append(fig)

        sp0 = subplot(sp_ctr, title='Image %i raw' % n_image, xlabel='x ($\mu$m)', ylabel='y ($\mu$m)')
        sp_ctr += 1
        sp1 = subplot(sp_ctr, title='Image %i processed' % n_image, xlabel='x ($\mu$m)', ylabel='y ($\mu$m)')
        sp_ctr += 1

        min_index_x = np.argwhere(x_axis0 == x_axis[0]).squeeze()
        max_index_x = np.argwhere(x_axis0 == x_axis[-1]).squeeze()

        min_index_y = np.argwhere(y_axis0 == y_axis[0]).squeeze()
        max_index_y = np.argwhere(y_axis0 == y_axis[-1]).squeeze()

        image_recover = image.copy()
        mask_dest = image == 0
        image_recover += bg[min_index_y:max_index_y+1,min_index_x:max_index_x+1]
        image_recover[mask_dest] = 0

        for im, sp in [(image, sp0), (image_recover, sp1)]:
            image_float = im.astype(np.float64)
            image_floored = image_float - np.median(image_float)
            image_floored[image_floored < 0] = 0
            image_normed = image_floored/np.max(image_floored)
            colors = cmap(image_normed)
            colors[mask_dest] = np.array([1, 0, 0, 1])

            sp.imshow(colors, extent=extent, aspect='auto')

    ms.saveall('./album_2021-05-18/%s' % os.path.basename(lasing_file))
    for fig in figs:
        fig.close()



