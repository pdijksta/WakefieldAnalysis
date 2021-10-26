import numpy as np
import itertools
import h5_storage
import image_and_profile as iap
import myplotstyle as ms

ms.closeall()

blmeas_file = '/mnt/data/data_2021-10-24/132878669_bunch_length_meas.h5'
blmeas_dict = h5_storage.loadH5Recursive(blmeas_file)

_dim_result = blmeas_dict['Raw data']['Dimension result']

_dim1 = n_phases = _dim_result[0]
_dim2 = n_images = _dim_result[1]
_dim3 = _dim_result[2]

mydata = []

#readables = ['bs://gr_y_fit_mean',
#             'bs://gr_y_fit_standard_deviation',
#             'bs://image',
#             'bs://x_axis',
#             'bs://y_axis',
#             'bs://gr_x_profile',
#             'bs://gr_y_profile',
#             'bs://gr_x_fit_gauss_function',
#             'bs://gr_y_fit_gauss_function',
#             'bs://gr_x_fit_standard_deviation',
#             'bs://gr_x_axis',
#             'bs://gr_y_axis']


for i in range(0, _dim1):
    li = []
    mydata.append(li)
    for j in range(0, _dim2):
        lj = []
        li.append(lj)
        for k in range(0, _dim3):
            lj.append(blmeas_dict['Raw data']['result%i%i%i' % (i,j,k)])

calibration = blmeas_dict['Processed data']['Calibration'] * 1e-6/1e-15
charge = 180e-12

current_profiles = []

for n_phase, n_image in itertools.product(range(n_phases), range(n_images)):
    image_data = mydata[n_phase][n_image][2].astype(np.float64)
    x_axis = mydata[n_phase][n_image][3].astype(np.float64)*1e6
    y_axis = mydata[n_phase][n_image][4].astype(np.float64)*1e6
    current = image_data.sum(axis=1)
    time_arr = y_axis / calibration
    current_profiles.append(iap.BeamProfile(time_arr, current, 6e9, charge))

ms.figure('All current profiles')

ms.show()

