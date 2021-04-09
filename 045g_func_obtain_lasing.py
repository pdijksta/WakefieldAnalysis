import numpy as np

from h5_storage import loadH5Recursive
import myplotstyle as ms
import data_loader
import elegant_matrix
import tracking
import image_and_profile as iap
import misc
import lasing

ms.plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant')

data_dir = '/mnt/data/data_2021-03-16/'
archiver_dir = '/mnt/data/archiver_api_data/'
lasing_on_file = data_dir+'20210316_202944_SARBD02-DSCR050_camera_snapshot.h5'
lasing_off_file = data_dir+'20210316_204139_SARBD02-DSCR050_camera_snapshot.h5'
cutoff = 0.03
cutoff_proj = 0.1
x0 = 0.0005552048387736093
n_slices = 50

streaker_data_file = archiver_dir+'2021-03-16_1.h5'
streaker_data = data_loader.DataLoader(file_h5=streaker_data_file)
streaker = 'SARUN18-UDCP020'

timestamp1 = elegant_matrix.get_timestamp(2021, 3, 16, 20, 29, 44)
timestamp2 = elegant_matrix.get_timestamp(2021, 3, 16, 20, 41, 39)

for timestamp in timestamp1, timestamp2:
    streaker_gap = streaker_data.get_prev_datapoint(streaker+':GAP', timestamp)*1e-3
    streaker_center = streaker_data.get_prev_datapoint(streaker+':CENTER', timestamp)*1e-3
    print('Streaker properties [mm]', timestamp, streaker_gap*1e3, streaker_center*1e3)

lasing_on = loadH5Recursive(lasing_on_file)
lasing_off = loadH5Recursive(lasing_off_file)

image_on = lasing_on['camera1']['image'].astype(float)
image_off = lasing_off['camera1']['image'].astype(float)

x_axis = lasing_on['camera1']['x_axis'].astype(float)*1e-6
y_axis = lasing_on['camera1']['y_axis'].astype(float)*1e-6

image_obj_on = iap.Image(image_on, x_axis, y_axis, subtract_median=True, x_offset=x0)
image_obj_off = iap.Image(image_off, x_axis, y_axis, subtract_median=True, x_offset=x0)

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

meas_screen = misc.image_to_screen(image_off, x_axis, True, x_offset=x0)
meas_screen.cutoff2(5e-2)
meas_screen.crop()
meas_screen.reshape(len_profile)

ms.figure('Obtain x-t')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x axis', ylabel='Intensity (arb. units)')
sp_ctr += 1

sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
sp_ctr += 1

sp_tx = subplot(sp_ctr, title='t-x', xlabel='t[fs]', ylabel='x [mm]')
sp_ctr += 1

meas_screen.plot_standard(sp_screen)

gaps = [streaker_gap, streaker_gap]
beam_offsets = [0, -(streaker_center - streaker_offset)]

gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge)

final_profile = gauss_dict['final_profile']
final_screen = gauss_dict['final_screen']

final_screen.plot_standard(sp_screen)
final_profile.plot_standard(sp_profile, center='Max')
final_profile._xx = final_profile._xx - final_profile._xx.min()

r12 = tracker.calcR12()[n_streaker]

wake_t, wake_x = final_profile.get_x_t(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker], r12)

t_axis = np.interp(x_axis, wake_x, wake_t)

sp_tx.plot((wake_t-wake_t.min())*1e15, wake_x*1e3)

dispersion = tracker.calcDisp()[1]

lasing_dict = lasing.obtain_lasing(image_obj_off, image_obj_on, n_slices, wake_x, wake_t, len_profile, dispersion, tracker.energy_eV, charge)
all_slice_dict = lasing_dict['all_slice_dict']
all_image_dict = lasing_dict['all_images']

fig = ms.figure('Backtracked images')
ms.plt.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1
for label, subdict in all_image_dict.items():
    image_cut = subdict['image_cut']
    image_tE = subdict['image_tE']

    sp = subplot(sp_ctr, title=label, xlabel='x [mm]', ylabel='y [mm]')
    sp_ctr += 1
    image_cut.plot_img_and_proj(sp, revert_x=True)

    sp = subplot(sp_ctr, title=label, xlabel='t [fs]', ylabel='$\Delta$ E [MeV]')
    sp_ctr += 1
    image_tE.plot_img_and_proj(sp)



ms.figure('Lasing reconstruction')
ms.plt.subplots_adjust(hspace=.35)
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_current = subplot(sp_ctr, title='Current', xlabel='t [fs]', ylabel='I [kA]')
sp_ctr += 1

sp_mean = subplot(sp_ctr, title='Slice energy', xlabel='t [fs]', ylabel='$\Delta$ E [MeV]')
sp_ctr += 1

sp_sigma = subplot(sp_ctr, title='Slice $\sigma$', xlabel='t [fs]', ylabel='Slice width [MeV]')
sp_ctr += 1

for label, slice_dict in all_slice_dict.items():
    slice_time = slice_dict['slice_x']
    slice_proj = slice_dict['slice_intensity']
    slice_current = slice_dict['slice_current']
    sp_current.plot(slice_time*1e15, slice_current/1e3, marker='.', label=label)

    sp_mean.plot(slice_dict['slice_x']*1e15, slice_dict['slice_mean']*1e-6, label=label, marker='.')
    sp_sigma.plot(slice_dict['slice_x']*1e15, slice_dict['slice_sigma']*1e-6, label=label, marker='.')

sp_lasing = subplot(sp_ctr, title='Lasing', xlabel='t [fs]', ylabel='P (GW)')
sp_ctr += 1

mean_current = (all_slice_dict['Lasing_off']['slice_current'] + all_slice_dict['Lasing_on']['slice_current'])/2.
mask_current = mean_current > mean_current.max()*0.1
mean_current[~mask_current] = 0

delta_E = all_slice_dict['Lasing_off']['slice_mean'] - all_slice_dict['Lasing_on']['slice_mean']
delta_std_sq = all_slice_dict['Lasing_on']['slice_sigma']**2 - all_slice_dict['Lasing_off']['slice_sigma']**2

np.clip(delta_E, 0, None, out=delta_E)
np.clip(delta_std_sq, 0, None, out=delta_std_sq)

slice_time = all_slice_dict['Lasing_off']['slice_x']

power_from_Eloss = lasing.power_Eloss(mean_current, delta_E)
E_total = np.trapz(power_from_Eloss, slice_time)
power_from_Espread = lasing.power_Espread(slice_time, mean_current, delta_std_sq, E_total)
E_total2 = np.trapz(power_from_Espread, slice_time)

sp_lasing.plot(slice_time*1e15, power_from_Eloss/1e9, label='Eloss', marker='.')
sp_lasing.plot(slice_time*1e15, power_from_Espread/1e9, label='Espread', marker='.')

print('Energy from Eloss: %i [uJ]' % (E_total*1e6))
print('Energy from Espread: %i [uJ]' % (E_total2*1e6))

for sp_ in sp_current, sp_mean, sp_sigma:
    sp_.legend()

ms.plt.show()

