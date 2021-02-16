import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname
from h5_storage import loadH5Recursive
import gaussfit

import tracking
import elegant_matrix

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

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
mean_offset = 0.472
beam_offsets = [0, 4.692e-3]
n_streaker = 1
tt_halfrange = 200e-15
bp_smoothen = 1e-15

emittance_arr = np.array([1., 200., 300., 400., 500.,])*1e-9

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

magnet_file = archiver_dir + 'archiver_api_data/2020-10-03.h5'

blmeas38 = dirname1+'129833611_bunch_length_meas.h5'


file0 = dirname1 + 'Passive_data_20201003T231958.mat'

dict0 = loadH5Recursive(file0+'.h5')

subtract_min = True

def get_screen_from_proj(projX, x_axis, invert_x):
    if invert_x:
        xx, yy = (-x_axis[::-1]).copy(), (projX[::-1]).copy()
    else:
        xx, yy = x_axis.copy(), projX.copy()
    if subtract_min:
        yy -= yy.min()
    screen = tracking.ScreenDistribution(xx, yy)
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen

invert_x0 = True
x_axis = dict0['x_axis'][::-1]*1e-6
projx = dict0['projx'][0]
projx0 = dict0['projx'][-1]
all_mean = []
for proj in projx0:
    screen = get_screen_from_proj(proj, x_axis, invert_x0)
    xx, yy = screen._xx, screen._yy
    gf = gaussfit.GaussFit(xx, yy)
    all_mean.append(gf.mean)

mean0 = np.mean(all_mean)


profile_meas = tracking.profile_from_blmeas(blmeas38, tt_halfrange, charge, energy_eV)
#profile_meas.flipx()

for n_proj in range(10):

    screen0 = get_screen_from_proj(projx[n_proj], x_axis, invert_x0)
    screen0._xx = screen0._xx - mean0
    screen0.cutoff(3e-2)
    screen0.crop()



    ms.figure('Fit quad effect')
    subplot = ms.subplot_factory(1,2)
    sp_ctr = 1

    sp_profile = subplot(sp_ctr, title='Current profile', xlabel='time [fs]', ylabel='I (arb. units)')
    sp_ctr += 1
    profile_meas.plot_standard(sp_profile)

    sp_forward = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    screen0.plot_standard(sp_forward, label='Measured', lw=3, color='black')

    tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, bp_smoothen=bp_smoothen)


    for emittance in emittance_arr:
        tracker.n_emittances = [emittance, n_emittances[1]]
        forward_dict = tracker.matrix_forward(profile_meas, gaps, beam_offsets)
        screen = forward_dict['screen_no_smoothen']
        screen.smoothen(smoothen)

        bs = forward_dict['bs_at_streaker'][1]

        screen.crop()
        screen.plot_standard(sp_forward, label='%i $\mu$m' % (bs*1e6))


    sp_forward.legend(title='RMS beamsize at streaker')


plt.show()

