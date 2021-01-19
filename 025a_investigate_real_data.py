import socket
import matplotlib.pyplot as plt
import numpy as np
import mat73
from scipy.constants import c

import tracking
import elegant_matrix

import myplotstyle as ms


charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 10, 3, 23, 19, 58)
struct_lengths = [1., 1.]
n_particles = int(50e3)
n_emittances = [300e-9, 300e-9]
screen_bins = 200
screen_cutoff = 1e-3
smoothen = 0
profile_cutoff = 0
len_profile = int(1e3)
gap2_correcting_summand = -40e-6
gaps = [10e-3, 10e-3+gap2_correcting_summand]
mean_struct2 = 472e-6 # see 026_script



plt.close('all')

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

file_ = dirname1 + '/Passive_data_20201003T231958.mat'


dict_ = mat73.loadmat(file_)


x_axis0 = dict_['x_axis'].astype(np.float64) * 1e-6
y_axis0 = dict_['y_axis'].astype(np.float64) * 1e-6

if np.diff(x_axis0)[0] < 0:
    x_axis = x_axis0[::-1]

if np.diff(y_axis0)[0] < 0:
    y_axis = y_axis0[::-1]


def get_image(i, j):
    image = dict_['Image'][i][j].T.astype(np.float64)
    if np.diff(x_axis0)[0] < 0:
        image = image[:,::-1]
    if np.diff(y_axis0)[0] < 0:
        image = image[::-1,:]
    return image


n_offset = 0
image = get_image(n_offset, 0)
image0 = get_image(-1, 0)


ms.figure('Investigate')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

screen = tracking.ScreenDistribution(x_axis.copy(), image.sum(axis=0))
screen0 = tracking.ScreenDistribution(x_axis.copy(), image0.sum(axis=0))

shift0 = float(screen0.gaussfit.mean)
screen._xx -= shift0
screen0._xx -= shift0


screen._yy -= screen._yy.min()
screen.reshape(1e3)
screen.cutoff(0.05)
screen.remove0()
screen.reshape(1e3)

sp_proj = subplot(sp_ctr, title='Screen', xlabel='Position [mm]', ylabel='Intensity (arb. units)')
sp_ctr += 1

screen.plot_standard(sp_proj, label='Streaking')
screen0.plot_standard(sp_proj, label='No streaking')

tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile)
r12 = tracker.calcR12()[1]

bp_gauss = tracking.get_gaussian_profile(30e-15, 200e-15, 1e3, charge, tracker.energy_eV)
offset_arr = dict_['value']*1e-3 - mean_struct2

wf = bp_gauss.calc_wake(gaps[1], offset_arr[n_offset], struct_lengths[1])

dipole_wake = wf['dipole']['wake_potential']
dipole_t = wf['input']['charge_xx']/c

sp_wf = subplot(sp_ctr, title='Wakefield', xlabel='Time [fs]', ylabel='Wake [V/m]')
sp_ctr += 1

sp_wf.plot(dipole_t, dipole_wake)

sp_deltax = subplot(sp_ctr, title='Wake effect', xlabel='Time [fs]', ylabel='$\Delta$ x')
sp_ctr += 1

deltax = dipole_wake * r12 / tracker.energy_eV

sp_deltax.plot(dipole_t, deltax)




plt.show()

