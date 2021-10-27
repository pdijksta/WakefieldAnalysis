import numpy as np
import socket
import h5_storage
import blmeas
import gaussfit
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

ms.figure('Blmeas')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

for n_zero_crossing, zc_string in enumerate(['', ' 2'], 1):
    images = blmeas_dict['Processed data']['Beam images'+zc_string]
    x_axis = blmeas_dict['Processed data']['x axis'+zc_string]
    y_axis = blmeas_dict['Processed data']['y axis'+zc_string]
    t_axis = y_axis / calibration

    curr_best, all_current = blmeas.get_average_blmeas_profile(images, x_axis, y_axis, calibration)
    current0 = images.astype(np.float64).sum(axis=-1)
    current = current0.reshape([current0.size//current0.shape[-1], current0.shape[-1]])

    sp = subplot(sp_ctr, title='Zero crossing %i' % n_zero_crossing, xlabel='t (fs)', ylabel='I (arb. units)')
    sp_ctr += 1

    for n_curr, curr in enumerate(list(current)+[curr_best]):
        yy = curr - curr.min()
        if curr is curr_best:
            color, lw = 'black', 3
        else:
            color, lw = None, None
        gf = gaussfit.GaussFit(t_axis, yy)
        xx = t_axis - gf.mean
        sp.plot(xx*1e15, yy, color=color, lw=lw)


ms.show()




