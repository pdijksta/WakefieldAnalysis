#import numpy as np
import glob
import socket
import os

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

blmeas_files = glob.glob(dir_+'*bunch_length_meas.h5')+glob.glob(dir_.replace('24', '18')+'*bunch_length_meas.h5')

for center in (True, False):

    for blmeas_file in blmeas_files:

        blmeas_dict0 = h5_storage.loadH5Recursive(blmeas_file)

        ms.figure(os.path.basename(blmeas_file), figsize=(14,10))
        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1

        blmeas_dict = blmeas.load_avg_blmeas(blmeas_dict0)

        for nz, zc_dict in blmeas_dict.items():
            t_axis = zc_dict['time']
            curr_best = zc_dict['current_reduced']
            all_current = zc_dict['all_current_reduced']

            sp = subplot(sp_ctr, title='Zero crossing %i' % (nz+1), xlabel='t (fs)', ylabel='I (arb. units)')
            sp_ctr += 1

            for n_curr, curr in enumerate(list(all_current)+[curr_best]):
                yy = curr - curr.min()
                if curr is curr_best:
                    color, lw = 'black', 3
                else:
                    color, lw = None, None
                if center:
                    gf = gaussfit.GaussFit(t_axis, yy)
                    xx = t_axis - gf.mean
                else:
                    xx = t_axis
                sp.plot(xx*1e15, yy, color=color, lw=lw)

            sp.set_xlim(xx.min()*1e15, xx.max()*1e15)


    ms.saveall('./blmeas_plots/071d_center_%s' % center, empty_suptitle=False, ending='.pdf')
    ms.closeall()


ms.show()

