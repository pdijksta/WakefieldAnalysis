import numpy as np
import mat73
from h5_storage import saveH5Recursive

files1 = [
        #'Passive_data_20201003T231812.mat',
        #'Passive_data_20201003T231958.mat',
        #'Passive_data_20201003T233852.mat',
        ]

files2 = [
        #'Passive_data_20201004T161118.mat',
        #'Passive_data_20201004T172425.mat',
        #'Passive_data_20201004T223859.mat',
        #'Passive_data_20201004T163828.mat',
        'Passive_data_20201004T221502.mat',
        #'Passive_money_20201004T012247.mat',
        #'Passive_data_20201004T163828.mat',
        ]

dirname1 = '/home/work/data_2020-10-03/'
dirname2 = '/home/work/data_2020-10-04/'


for dir_, files, in [(dirname1, files1), (dirname2, files2)]:
    for file_ in files:
        outp_dict = {}
        inp_dict = mat73.loadmat(dir_+file_)
        print('Image' in inp_dict)
        for key in ['Image', 'knob', 'value', 'x_axis', 'y_axis']:
            try:
                value = inp_dict[key]
            except KeyError:
                print('Continue for key %s' % key)
                continue
            if key == 'Image':
                outp = []
                if type(value[0]) is list:
                    for n_offset in range(len(value)):
                        outp.append([])
                        for n_image in range(len(value[n_offset])):
                            img = value[n_offset][n_image].T
                            outp[n_offset].append(img.sum(axis=0))
                else:
                    for n_image in range(len(value)):
                        img = value[n_image].T
                        outp.append(img.sum(axis=0))
                outp_dict['projx'] = np.array(outp)
            else:
                outp_dict[key] = np.array(value)


        new_file = dir_+file_+'.h5'
        saveH5Recursive(new_file, outp_dict)
        print(new_file)




