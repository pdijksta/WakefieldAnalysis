import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c
import data_loader

import myplotstyle as ms

plt.close('all')


data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'

meas_charge = [
        ('Bunch_length_meas_2020-02-03_21-54-24.h5', 200),
        ('Bunch_length_meas_2020-02-03_22-07-39.h5', 200),
        ('Bunch_length_meas_2020-02-03_22-08-38.h5', 200),
        ('Bunch_length_meas_2020-02-03_23-55-33.h5', 100),
        ]

ms.figure('All blmeas of 03.02.2020')
sp_ctr = 1
subplot = ms.subplot_factory(2,2)

sp_current = subplot(sp_ctr, title='Current 1', xlabel='z [$\mu$m]', ylabel='I [kA]')
sp_ctr += 1


sp_current2 = subplot(sp_ctr, title='Current 2')
sp_ctr += 1


for ctr, (blmeas, charge) in enumerate(meas_charge):
    label = ctr
    meas = data_loader.load_blmeas(data_dir + blmeas)

    xx = meas['time_profile1']*c
    yy = meas['current1']

    sp_current.plot(xx*1e6, yy/1e3, label=label)
    print(ctr, '%.1e' % np.trapz(yy, xx))

    if meas['time_profile2'] is not None:

        xx = meas['time_profile2']*c
        yy = meas['current2']

        sp_current2.plot(xx*1e6, yy/1e3, label=label)

xlim = sp_current.get_xlim()
sp_current2.set_xlim(*xlim)

sp_current.legend()
sp_current2.legend()

plt.show()

