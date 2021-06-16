from collections import OrderedDict
import numpy as np
import pickle
import matplotlib.pyplot as plt

from h5_storage import loadH5Recursive
import image_and_profile as iap
import tracking
import elegant_matrix
import misc2 as misc

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

#data_dir = '/mnt/data/data_2021-03-16/'
data_dir = '/sf/data/measurements/2021/03/16/'
x0 = 0.0005552048387736093
profile_file = './backtrack_image_no_compensate.pkl'
with open(profile_file, 'rb') as f:
    wake_dict = pickle.load(f)

xx = wake_dict['xx']


lasing_on_file = data_dir+'20210316_202944_SARBD02-DSCR050_camera_snapshot.h5'
lasing_off_file = data_dir+'20210316_204139_SARBD02-DSCR050_camera_snapshot.h5'

lasing_on = loadH5Recursive(lasing_on_file)
lasing_off = loadH5Recursive(lasing_off_file)

image_on = lasing_on['camera1']['image'].astype(float)
image_off = lasing_off['camera1']['image'].astype(float)

x_axis = lasing_on['camera1']['x_axis'].astype(float)*1e-6
y_axis = lasing_on['camera1']['y_axis'].astype(float)*1e-6

if x_axis[1] < x_axis[0]:
    x_axis = x_axis[::-1]
    image_on = image_on[:,::-1]
    image_off = image_off[:,::-1]

if y_axis[1] < y_axis[0]:
    y_axis = y_axis[::-1]
    image_on = image_on[::-1,:]
    image_off = image_off[::-1,:]

image_on -= np.median(image_on)
image_off -= np.median(image_off)

np.clip(image_on, 0, None, out=image_on)
np.clip(image_off, 0, None, out=image_off)

image_on = iap.Image(image_on, x_axis, y_axis, x_offset=x0)
image_off = iap.Image(image_off, x_axis, y_axis, x_offset=x0)

ms.figure('')
subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1

all_slice_dict = OrderedDict()

for image, label in [(image_on, 'Lasing On'), (image_off, 'Lasing Off')][::-1]:
    image_cut = image.cut(xx.min(), 0.5e-3)
    sp = subplot(sp_ctr, xlabel='x [mm]', ylabel='y [mm]', title=label)
    sp_ctr += 1

    image_cut.plot_img_and_proj(sp)
    n_slices = 20
    image_slice = image_cut.slice_x(n_slices)
    slice_dict = image_slice.fit_slice()

    slice_mean = slice_dict['slice_mean']
    slice_std = slice_dict['slice_sigma']
    mask = slice_std*2 < (image_slice.y_axis.max() - image_slice.y_axis.min())
    proj_sliced = image_slice.image.sum(axis=-2)
    mask2 = proj_sliced > proj_sliced.max()*0.05
    mask = np.logical_and(mask, mask2)
    sp.errorbar(image_slice.x_axis[mask]*1e3, slice_mean[mask]*1e3, yerr=slice_std[mask]*1e3, ls='None', marker='+', color='red')

    all_slice_dict[label] = {
            'mean': slice_mean[mask],
            'std': slice_std[mask],
            'slice_x': image_slice.x_axis[mask],
            'proj_x': image_cut.x_axis,
            'proj': image_cut.image.sum(axis=-2),
            }

sp_mean = subplot(sp_ctr, title='Slice centroids', xlabel='x [mm]', ylabel='Centroid position [mm]')
sp_ctr += 1

sp_sigma = subplot(sp_ctr, title='Slice widths', xlabel='x [mm]', ylabel='Slice width [mm]')
sp_ctr += 1

#sp_current = subplot(sp_ctr, title='Projected intensity', xlabel='x [mm]', ylabel='Intensity (arb. units)')
#sp_ctr += 1

for ctr, (label, subdict) in enumerate(all_slice_dict.items()):
    xx = subdict['slice_x']
    mean = subdict['mean']
    std = subdict['std']

    diff_x = xx[1] - xx[0]

    sp_mean.plot(xx*1e3, mean*1e3, label=label)
    sp_sigma.plot(xx*1e3, std*1e3, label=label)

    xx = subdict['proj_x']
    yy = subdict['proj']
    #sp_current.plot(xx*1e3, yy/1e5, label=label)

for sp_ in sp_mean, sp_sigma:
    sp_.legend()

plot_gaussfits = False
sp_ctr = np.inf
ny, nx = 3, 3
subplot = ms.subplot_factory(ny, nx, grid=False)
if plot_gaussfits:
    for n_slice in range(n_slices):
        gf = slice_dict['slice_gf'][n_slice]

        if sp_ctr > ny*nx:
            ms.figure('Gaussfit details')
            sp_ctr = 1
        sp = subplot(sp_ctr, xlabel='y axis', ylabel='Slice intensity', title='Slice %i' % n_slice)
        sp_ctr += 1

        sp.plot(gf.xx, gf.yy)
        sp.plot(gf.xx, gf.reconstruction)





ms.figure('Reconstruction')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

charge = 200e-12
len_profile = int(2e3)

sp_profile = subplot(sp_ctr, title='Current profiles', xlabel='t [fs]', ylabel='Current [kA]')
sp_ctr += 1

sp_screen = subplot(sp_ctr, title='Screen projections', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr += 1



tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 2e-3
profile_cutoff = 2e-2
len_profile = int(2e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 30e-6
n_emittances = [500e-9, 500e-9]
n_particles = int(100e3)
n_streaker = 1
self_consistent = True
quad_wake = False
bp_smoothen = 1e-15
invert_offset = True
#sig_t_range = np.arange(20, 40.01, 2)*1e-15

#mean_struct2 = 472e-6 # see 026_script
fudge_factor = 30e-6
mean_struct2 = 466e-6 + fudge_factor
gap2_correcting_summand = 0 #-3e-6
sig_t_range = np.arange(20, 40.01, 5)*1e-15
gaps = [10e-3, 10e-3]
subtract_min = True
fit_emittance = True
archiver_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/'
magnet_file = archiver_dir + 'archiver_api_data/2021-03-16.h5'
timestamp = elegant_matrix.get_timestamp(2021, 3, 16, 20, 41, 39)
beam_offsets = [-0.0, -0.004724]

median_screen = misc.image_to_screen(image_on.image, x_axis, False, x0)
median_screen.cutoff2(3e-2)
median_screen.crop()
median_screen.reshape(len_profile)

tracker = tracking.Tracker(magnet_file=magnet_file, timestamp=timestamp, n_particles=n_particles, n_emittances=n_emittances, screen_bins=screen_bins, screen_cutoff=screen_cutoff, smoothen=smoothen, profile_cutoff=profile_cutoff, len_screen=len_profile, bp_smoothen=bp_smoothen, quad_wake=False)

energy_eV = tracker.energy_eV
blmeas = data_dir+'113876237_bunch_length_meas.h5'

profile_meas = iap.profile_from_blmeas(blmeas, 200e-15, charge, energy_eV)
profile_meas.cutoff(5e-2)
profile_meas.reshape(len_profile)
profile_meas.crop()
tt_range = (profile_meas.time.max() - profile_meas.time.min())

profile_gauss = iap.get_gaussian_profile(38e-15, tt_range, len_profile, charge, energy_eV)



bp_back = tracker.track_backward2(median_screen, profile_gauss, gaps, beam_offsets, 1)

bp_back2 = tracker.track_backward2(median_screen, bp_back, gaps, beam_offsets, 1)

#bp_back3 = tracker.track_backward2(median_screen, bp_back2, gaps, beam_offsets, 1)

#bp_back4 = tracker.track_backward2(median_screen, bp_back3, gaps, beam_offsets, 1)

median_screen.plot_standard(sp_screen, label='Measured', color='black')
def get_color(label):
    if 'measured' in label:
        return 'black'
    elif 'Back 1' in label:
        return 'green'
    elif 'Back 2' in label:
        return 'blue'
    elif 'Gaussian' in label:
        return 'orange'
    elif 'TDC forward' in label:
        return 'grey'
    else:
        return None

for bp, label in [
        #(profile_meas, 'TDC forward'),
        (bp_back, 'Back 1 forward'),
        (bp_back2, 'Back 2 forward'),
        #(bp_back3, 'Back 3 forward'),
        #(bp_back4, 'Back 4 forward'),
        ]:
    color = get_color(label)
    forward_dict = tracker.elegant_forward(bp, gaps, beam_offsets)
    screen = forward_dict['screen']
    screen.plot_standard(sp_screen, label=label, color=color)

for profile, label in [
        (profile_meas, 'TDC measured'),
        (profile_gauss, 'Gaussian'),
        (bp_back, 'Back 1, using Gauss'),
        (bp_back2, 'Back 2, using 1'),
        #(bp_back3, 'Back 3, using 2'),
        #(bp_back4, 'Back 4, using 3'),
        ]:
    color = get_color(label)
    if profile is profile_meas:
        ls = '--'
    else:
        ls = None
    profile.plot_standard(sp_profile, label=label, color=color, ls=ls)

sp_profile.legend()
sp_screen.legend()

sp_screen.set_xlim(-2.2, 0.3)

ms.saveall('/tmp/for_spie', hspace=0.35, ending='.pdf', trim=False)

plt.show()

