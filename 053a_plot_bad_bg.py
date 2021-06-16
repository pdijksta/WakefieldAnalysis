import pickle
import matplotlib.pyplot as plt
import numpy as np

import h5_storage
import myplotstyle as ms

plt.close('all')

snapshot_file = '/sf/data/measurements/2021/05/19/20210519_121718_SARBD02-DSCR050_camera_snapshot.h5'

image_dict = h5_storage.loadH5Recursive(snapshot_file)

x_axis = image_dict['camera1']['x_axis']
y_axis = image_dict['camera1']['y_axis']
image = image_dict['camera1']['image']

with open('./bytes.pkl', 'rb') as f:
    bytes = pickle.load(f)
arr0 = np.frombuffer(bytes, dtype=np.uint16)
arr = arr0.reshape([2160, 2560])


ms.figure('Background')
sp = ms.subplot_factory(1,1, False)(1)

sp.imshow(arr, aspect='auto')

ms.figure('Saved')
sp = ms.subplot_factory(1,1, False)(1)

sp.imshow(image, aspect='auto')

plt.show()

