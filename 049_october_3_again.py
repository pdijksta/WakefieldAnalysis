import matplotlib.pyplot as plt
import itertools
import numpy as np
import mat73
import h5_storage
#from scipy.io import loadmat
from socket import gethostname

import analysis

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2020-10-03/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2020/10/03/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-10-03/'

data_dir2 = data_dir.replace('03', '04')

#matfiles = [
#        data_dir + 'Passive_data_20201003T231812.mat',
#        data_dir + 'Passive_data_20201003T231958.mat',
#        data_dir + 'Passive_data_20201003T233852.mat',
#        data_dir2 + 'Passive_data_20201004T161118.mat',
#        data_dir2 + 'Passive_data_20201004T172425.mat',
#        data_dir2 + 'Passive_data_20201004T223859.mat',
#        data_dir2 + 'Passive_data_20201004T163828.mat',
#        data_dir2 + 'Passive_data_20201004T221502.mat',
#        ]
#
#for mat in matfiles:
#    try:
#        dict_ = loadmat(mat)
#    except NotImplementedError:
#        dict_ = mat73.loadmat(mat)
#    if 'knob' in dict_:
#        print(mat, dict_['knob'], dict_['value'])
#    else:
#        print('No knob in %s' % mat)


matfile = data_dir2 + 'Passive_data_20201004T172425.mat'

dict_ = mat73.loadmat(matfile)

new_offsets = dict_['value']*1e-3
streaker = dict_['knob'].split(':')[0]
screen = 'SARBD02-DSCR050'
x_axis = dict_['x_axis']*1e-6
gap_mm = 10
projections = np.zeros([len(dict_['Image']), len(dict_['Image'][0]), len(x_axis)], dtype=np.float32)

for a, b in itertools.product(range(len(dict_['Image'])), range(len(dict_['Image'][0]))):
    projections[a,b] = dict_['Image'][a][b].sum(axis=-1)



new_dict = {
        'streaker_offsets': new_offsets,
        'screen': screen,
        'n_images': projections.shape[1],
        'streaker': streaker,
        'pyscan_result': {
            'x_axis_m': x_axis,
            'projx': projections,
            },
        'meta_data': {
            streaker+':GAP': gap_mm,
            },
        }

h5_storage.saveH5Recursive(matfile.replace('.mat', '_new.h5'), new_dict)


streaker_analysis = analysis.analyze_streaker_calibration(new_dict)


plt.show()

