import socket
import matplotlib.pyplot as plt

import tracking

import myplotstyle as ms

plt.close('all')

tt_halfrange = 200e-15
charge = 200e-12


hostname = socket.gethostname()

# 3rd October
if hostname == 'desktop':
    dirname1 = '/storage/data_2020-10-03/'
    dirname2 = '/storage/data_2020-10-04/'
    archiver_dir = '/storage/Philipp_data_folder/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2020/10/03/'
    dirname2 = '/sf/data/measurements/2020/10/04/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2020-10-03/'
    dirname2 = '/home/work/data_2020-10-04/'
    archiver_dir = '/home/work/'

files1 = [
        #'Passive_data_20201003T231812.mat',
        'Passive_data_20201003T231958.mat',
        'Passive_data_20201003T233852.mat',]

files2 = [
        'Passive_data_20201004T161118.mat',
        'Passive_data_20201004T172425.mat',
        'Passive_data_20201004T223859.mat',
        'Passive_data_20201004T163828.mat',
        'Passive_data_20201004T221502.mat',
        #'Passive_money_20201004T012247.mat',
        ]
#blmeas_1 = dirname1+'Bunch_length_meas_2020-10-03_15-43-29.h5'
blmeas_1 = dirname1+'129833611_bunch_length_meas.h5'
blmeas_2 = dirname2+'129858802_bunch_length_meas.h5'

ms.figure('Measured beam profiles')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Beam profiles', xlabel='time [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

for blmeas in (blmeas_1, blmeas_2):
    for zero_crossing in (1, 2):
        bp = tracking.profile_from_blmeas(blmeas, tt_halfrange, charge, energy_eV=1, subtract_min=True, zero_crossing=zero_crossing)
        if zero_crossing == 2:
            bp.flipx()
        label = '%s %i' % (blmeas, zero_crossing)
        bp.plot_standard(sp, label=label)

sp.legend()



plt.show()

