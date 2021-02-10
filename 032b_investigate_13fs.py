import matplotlib.pyplot as plt
from socket import gethostname
from h5_storage import loadH5Recursive
from mat73 import loadmat; loadmat
import tracking

import myplotstyle as ms

plt.close('all')

len_profile = int(5e3)
charge = 200e-12
energy_eV = 4.5e9
struct_lengths = [1., 1.]
n_particles = int(1e5)
n_emittances = [500e-9, 500e-9]
screen_bins = 500
screen_cutoff = 1e-2
smoothen = 30e-6
profile_cutoff = 0
timestamp = 1601761132
gaps = [10e-3, 10e-3]
mean_struct2 = 472e-6 # see 026_script
beam_offsets38 = [0, 4.692e-3]
#beam_offsets25 = [0, 5.11e-3-mean_struct2]
n_streaker = 1
tt_halfrange = 200e-15
bp_smoothen = 1e-15



hostname = gethostname()
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


blmeas13 = dirname2 + '129920069_bunch_length_meas.h5'
file13 = dirname2 + 'Passive_data_20201004T221502.mat'
dict13 = loadmat(file13)
dict13_h5 = loadH5Recursive(file13+'.h5')

ms.figure('Short bunch')
subplot = ms.subplot_factory(1, 2)

sp_profile = subplot(1, title='Measured current profile')

sp_screen = subplot(2, title='Measured screen dist')


profile_meas = tracking.profile_from_blmeas(blmeas13, tt_halfrange, charge, energy_eV)

profile_meas.plot_standard(sp_profile)

x_axis = (dict13['x_axis']*1e-6)[::-1]
img0 = dict13['Image'][0][0]
proj0 = img0.T.sum(axis=0)


img1 = dict13['Image'][1][0]
proj1 = img1.T.sum(axis=0)

sp_screen.plot(x_axis*1e3, proj0, label='0')
sp_screen.plot(x_axis*1e3, proj1, label='1')

sp_screen.legend()




plt.show()





