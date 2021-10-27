import copy
import socket
import itertools
import numpy as np
import h5_storage
import image_and_profile as iap
import myplotstyle as ms

ms.closeall()

hostname = socket.gethostname()
if hostname == 'desktop':
    dir_ = '/storage/data_2021-10-24/'
elif hostname == 'pc11292.psi.ch':
    dir_ = '/sf/data/measurements/2021/10/24/'
elif hostname == 'pubuntu':
    dir_ = '/mnt/data/data_2021-10-24/'


blmeas_file = dir_+'132878669_bunch_length_meas.h5'
blmeas_dict = h5_storage.loadH5Recursive(blmeas_file)


calibration = blmeas_dict['Processed data']['Calibration'] * 1e-6/1e-15

charge = 180e-12
energy_eV = 6e9


for n_zero_crossing, zc_string in enumerate(['', ' 2'], 1):

    current_profiles = []
    current_profiles0 = []
    rms_arr = []

    x_axis = blmeas_dict['Processed data']['x axis']*1e-6
    y_axis = blmeas_dict['Processed data']['y axis']*1e-6
    n_phases, n_images, len_y, len_x = blmeas_dict['Processed data']['Beam images'+zc_string].shape

    for n_phase, n_image in itertools.product(range(n_phases), range(n_images)):

        image_data = blmeas_dict['Processed data']['Beam images'+zc_string][n_phase][n_image]
        current = image_data.sum(axis=1)
        time_arr = y_axis / calibration
        if time_arr[1] < time_arr[0]:
            time_arr = time_arr[::-1]
            current = current[::-1]
        bp0 = iap.BeamProfile(time_arr, current, 6e9, charge)
        current_profiles0.append(bp0)
        bp0 = copy.deepcopy(bp0)
        bp0.center('Mean')
        mask = np.logical_and(bp0.time > -100e-15, bp0.time < 100e-15)
        bp = iap.BeamProfile(time_arr[mask], current[mask], energy_eV, charge)
        bp.center('Mean')
        bp.cutoff2(3e-2)
        bp._yy -= bp._yy.min()
        bp.cutoff2(3e-2)
        bp.crop()
        bp.center('Mean')
        current_profiles.append(bp)
        rms_arr.append(bp.rms())

    rms_arr = np.array(rms_arr)

    ms.figure('All current profiles zero crossing %i' % n_zero_crossing, figsize=(12,10))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1
    sp = subplot(sp_ctr, title='Zero crossing %i' % n_zero_crossing, xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1

    for bp in current_profiles:
        bp.plot_standard(sp)

    n_sigma = 3
    n_points = 1000

    min_time, max_time = np.inf, -np.inf


    for bp in current_profiles0:
        bp._yy -= bp._yy.min()
        bp.center()
        gf = bp.gaussfit
        mask = np.logical_and(bp.time > gf.mean - n_sigma*gf.sigma, bp.time < gf.mean + n_sigma*gf.sigma)
        bp._xx = bp._xx[mask]
        bp._yy = bp._yy[mask]

        min_time = min(min_time, bp.time.min())
        max_time = max(max_time, bp.time.max())

    sp = sp1 = subplot(sp_ctr, title='Zero crossing 1 alt', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    for bp in current_profiles0:
        bp.plot_standard(sp)

    new_time = np.linspace(min_time, max_time, n_points)
    current_profiles1 = []
    for bp in current_profiles0:
        new_current = np.interp(new_time, bp.time, bp.current, left=0., right=0.)
        current_profiles1.append(iap.BeamProfile(new_time, new_current, energy_eV, charge))

    sp = sp2 = subplot(sp_ctr, title='Zero crossing 1 alt alt', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    for bp in current_profiles1:
        bp.plot_standard(sp)



    squares_mat = np.zeros([len(current_profiles0)]*2, float)

    for n_row in range(len(squares_mat)):
        for n_col in range(n_row):
            bp1 = current_profiles1[n_row]
            bp2 = current_profiles1[n_col]
            squares_mat[n_row,n_col] = squares_mat[n_col,n_row] = np.mean((bp1.current - bp2.current)**2)

    squares = squares_mat.sum(axis=1)
    n_best = np.argmin(squares)
    n_worst = np.argmax(squares)

    sp = subplot(sp_ctr, title='Best and worst', xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1

    current_profiles1[n_best].plot_standard(sp, label='Best')
    current_profiles1[n_worst].plot_standard(sp, label='Worst')


    for sp in sp1, sp2:
        current_profiles1[n_best].plot_standard(sp, label='Best', color='black', lw=3)

    sp.legend()





ms.show()

