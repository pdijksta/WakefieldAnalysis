import numpy as np
import mat73
import h5_storage

dirname1 = '/home/work/data_2020-10-03/'
dirname2 = '/home/work/data_2020-10-04/'

files = [
        'Passive_data_20201003T231958.mat',
        'Passive_data_20201003T233852.mat',
        'Passive_data_20201004T161118.mat',
        'Passive_data_20201004T172425.mat',
        'Passive_data_20201004T223859.mat',
        ]


for file_ in files:
    try:
        file1 = dirname1+file_
        dict_ = mat73.loadmat(file1)
    except:
        file1 = dirname2+file_
        dict_ = mat73.loadmat(file1)

    dict2 = {}
    for x,y in dict_.items():
        try:
            y2 = np.array(y)
            if y2.dtype != np.dtype('O'):
                dict2[x] = y2
        except:
            print('Pass for %s' % x)

    file2 = file1+'.h5'
    h5_storage.saveH5Recursive(file2, dict2)
    print('Saved %s' % file2)

