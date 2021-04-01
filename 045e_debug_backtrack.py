import numpy as np

import tracking
import image_and_profile as iap
from h5_storage import loadH5Recursive
import elegant_matrix
import misc
import myplotstyle as ms

ms.plt.close('all')


archiver_dir = '/mnt/data/archiver_api_data/'
data_dir = '/mnt/data/data_2021-03-16/'
x0 = 0.0005552048387736093

lasing_on_file = data_dir+'20210316_202944_SARBD02-DSCR050_camera_snapshot.h5'
lasing_off_file = data_dir+'20210316_204139_SARBD02-DSCR050_camera_snapshot.h5'

lasing_on = loadH5Recursive(lasing_on_file)
lasing_off = loadH5Recursive(lasing_off_file)

x_axis = lasing_on['camera1']['x_axis'].astype(float)*1e-6 - x0
y_axis = lasing_on['camera1']['y_axis'].astype(float)*1e-6

image_on = lasing_on['camera1']['image'].astype(float)
image_off = lasing_off['camera1']['image'].astype(float)

if x_axis[1] < x_axis[0]:
    x_axis = x_axis[::-1]
    image_on = image_on[:,::-1]
    image_off = image_off[:,::-1]

if y_axis[1] < y_axis[0]:
    y_axis = y_axis[::-1]
    image_on = image_on[::-1,:]
    image_off = image_off[::-1,:]




tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 2e-3
profile_cutoff = 2e-2
len_profile = int(2e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 30e-6
n_emittances = [900e-9, 500e-9]
n_particles = int(100e3)
n_streaker = 1
self_consistent = True
quad_wake = False
bp_smoothen = 1e-15
invert_offset = True
magnet_file = archiver_dir+'2021-03-16.h5'
timestamp = elegant_matrix.get_timestamp(2021, 3, 16, 20, 14, 10)
sig_t_range = np.arange(20, 50.01, 5)*1e-15
n_streaker = 1
compensate_negative_screen = False


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake, bp_smoothen=bp_smoothen, compensate_negative_screen=compensate_negative_screen)

streaker_offset = 0.00037758839957521145

meas_screen = misc.image_to_screen(image_off, x_axis, True, x_offset=0)
meas_screen.cutoff2(5e-2)
meas_screen.crop()
meas_screen.reshape(len_profile)

#ms.figure('Obtain x-t')
#subplot = ms.subplot_factory(2,2)
#sp_ctr = 1
#
#sp_screen = subplot(sp_ctr, title='Screen', xlabel='x axis', ylabel='Intensity (arb. units)')
#sp_ctr += 1
#
#
#sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
#sp_ctr += 1
#
#sp_tx = subplot(sp_ctr, title='t-x', xlabel='t[fs]', ylabel='x [mm]')
#sp_ctr += 1


#meas_screen.plot_standard(sp_screen)

streaker_gap = 0.01
streaker_center = 0.00512
gaps = [streaker_gap, streaker_gap]
beam_offsets = [0, -(streaker_center - streaker_offset)]


# Backtrack with Module
gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge)
final_profile = gauss_dict['final_profile']

module_profile = tracker.track_backward2(meas_screen, final_profile, gaps, beam_offsets, 1, plot_details=True)

# Backtrack manual

wake_dict = final_profile.calc_wake(streaker_gap, beam_offsets[1], 1)
r12 = tracker.calcR12()[1]
wake_effect = final_profile.wake_effect_on_screen(wake_dict, r12)
xx = wake_effect['x']
tt = wake_effect['t']

if xx[1] < xx[0]:
    xx = xx[::-1]
    tt = tt[::-1]

proj = meas_screen.intensity
x_axis = meas_screen.x

long_x_axis = np.linspace(x_axis[0], x_axis[-1], int(100e3))
long_proj = np.interp(long_x_axis, x_axis, proj)

t_interp0 = np.interp(long_x_axis, xx, tt)
intensity, bins = np.histogram(t_interp0, bins=100, weights=long_proj)
new_axis = (bins[1:] + bins[:-1])/2.
intensity[0] = 0
intensity[-1] = 0

manual_profile = iap.BeamProfile(new_axis, intensity, final_profile.energy_eV, 200e-12)
manual_profile_rev = iap.BeamProfile(new_axis, intensity[::-1], final_profile.energy_eV, 200e-12)

ms.figure('Debug')

subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]')
sp_ctr += 1

sp_screen.plot(meas_screen.x*1e3, meas_screen.intensity)

sp_wake = subplot(sp_ctr, title='Wake', xlabel='t [fs]', ylabel='x [mm]')
sp_ctr += 1

sp_wake.plot(tt*1e15, xx*1e3)

sp_profile = subplot(sp_ctr, title='Profile')
sp_ctr += 1

final_profile.plot_standard(sp_profile, label='Gauss opt', center='Gauss')
module_profile.plot_standard(sp_profile, label='Module', center='Gauss')
manual_profile.plot_standard(sp_profile, label='Manual', center='Gauss')

sp_profile.legend()

ms.plt.show()

