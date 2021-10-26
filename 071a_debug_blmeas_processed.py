import numpy as np; np
import itertools
import h5_storage
import image_and_profile as iap
import myplotstyle as ms

ms.closeall()

blmeas_file = '/mnt/data/data_2021-10-24/132878669_bunch_length_meas.h5'
blmeas_dict = h5_storage.loadH5Recursive(blmeas_file)

n_phases, n_images, len_y, len_x = blmeas_dict['Processed data']['Beam images'].shape

calibration = blmeas_dict['Processed data']['Calibration'] * 1e-6/1e-15
x_axis = blmeas_dict['Processed data']['x axis']*1e-6
y_axis = blmeas_dict['Processed data']['y axis']*1e-6

charge = 180e-12

current_profiles = []
rms_arr = []

for n_phase, n_image in itertools.product(range(n_phases), range(n_images)):

    image_data = blmeas_dict['Processed data']['Beam images'][n_phase][n_image]
    current = image_data.sum(axis=1)
    time_arr = y_axis / calibration
    if time_arr[1] < time_arr[0]:
        time_arr = time_arr[::-1]
        current = current[::-1]
    bp0 = iap.BeamProfile(time_arr, current, 6e9, charge)
    bp0.center('Mean')
    mask = np.logical_and(bp0.time > -100e-15, bp0.time < 100e-15)
    bp = iap.BeamProfile(time_arr[mask], current[mask], 6e9, charge)
    bp.center('Mean')
    bp.cutoff2(3e-2)
    bp._yy -= bp._yy.min()
    bp.cutoff2(3e-2)
    bp.crop()
    bp.center('Mean')
    current_profiles.append(bp)
    rms_arr.append(bp.rms())

rms_arr = np.array(rms_arr)

ms.figure('All current profiles', figsize=(12,10))
subplot = ms.subplot_factory(2,2)
sp_ctr = 1
sp = subplot(sp_ctr, title='Zero crossing 1', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

for bp in current_profiles:
    bp.plot_standard(sp)


ms.show()

