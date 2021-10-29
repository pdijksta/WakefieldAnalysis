import copy
import numpy as np
import argparse
from scipy.constants import c
from socket import gethostname

import lasing
import analysis
import tracking
import h5_storage
import config
import streaker_calibration
import image_and_profile as iap
import myplotstyle as ms
import elegant_matrix

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--save', type=str)
parser.add_argument('--recon-gap', action='store_true')
args = parser.parse_args()

ms.closeall()

config.fontsize=9

charge = 180e-12

title_fs = config.fontsize
ms.set_fontsizes(title_fs)
iap.absolute_ScreenProfile = True

elegant_matrix.set_tmp_dir('~/tmp_elegant/')


hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')

blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'

sc_file = data_dir1+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
sc_dict = h5_storage.loadH5Recursive(sc_file)
n_streaker = 1
#recon_gap = args.recon_gap

plot_gap_recon = True

if plot_gap_recon:
    recon_gap = True

gauss_kwargs = config.get_default_gauss_recon_settings()
gauss_kwargs['charge'] = charge
tracker_kwargs = config.get_default_tracker_settings()

blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'
blmeas_profile = iap.profile_from_blmeas(blmeas_file, gauss_kwargs['tt_halfrange'], gauss_kwargs['charge'], 0, True)
blmeas_profile.cutoff2(0.03)
blmeas_profile.crop()
blmeas_profile.reshape(1000)

sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3, charge)
for scf in (data_dir1+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'):
    sc.add_file(scf)

sc.fit_type('centroid')

tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
recon_kwargs['charge'] = charge
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(sc.meta_data)

offset_dict = sc.fit_type('centroid')
streaker_offset = offset_dict['streaker_offset']
index = -1
meas_screen = sc.get_meas_screens()[index]
meas_screen.cutoff2(tracker.screen_cutoff)
meas_screen.crop()
meas_screen.reshape(tracker.len_screen)


if recon_gap:
    gap_arr = np.array([10e-3-100e-6, 10e-3+0e-6])
    use_offsets = [0, 1, 2, 3, 12, 13, 14, 15]
    gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, recon_kwargs, streaker_offset, gap0=10e-3, use_offsets=use_offsets)
    print('assumed_bunch_duration %.2f' % (gap_recon_dict['beamsize']*1e15))
    print('assumed_bunch_uncertainty %.2f' % (gap_recon_dict['beamsize_rms']*1e15))
    delta_gap = gap_recon_dict['gap'] - 10e-3

    gap_arr = gap_recon_dict['gap_arr']
    beamsize_arr = gap_recon_dict['all_rms'].mean(axis=1)
    beamsize_plus = gap_recon_dict['beamsize'] + gap_recon_dict['beamsize_rms']
    beamsize_minus = gap_recon_dict['beamsize'] - gap_recon_dict['beamsize_rms']
    sort = np.argsort(gap_arr)
    gap_plus = np.interp(beamsize_plus, beamsize_arr[sort], gap_arr[sort])
    gap_minus = np.interp(beamsize_minus, beamsize_arr[sort], gap_arr[sort])
    print('Gap plus / minus, %.2f, %.2f' % (gap_plus*1e6, gap_minus*1e6))


else:
    delta_gap = -63e-6
print('Delta gap %i um' % (delta_gap*1e6))


tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(sc.meta_data)

recon_kwargs['gaps'] = [10e-3, 10e-3+delta_gap]
recon_kwargs['beam_offsets'] = [0., -(sc.offsets[index] - streaker_offset)]
recon_kwargs['n_streaker'] = 1
recon_kwargs['meas_screen'] = meas_screen
recon_kwargs['charge'] = charge


hspace, wspace = 0.40, 0.35
fig = ms.figure('Current profile reconstruction', figsize=(13, 6))
ms.plt.subplots_adjust(hspace=hspace, wspace=wspace)
subplot = ms.subplot_factory(2, 4, grid=False)
sp_ctr = 1


where0 = np.argwhere(sc.offsets == 0).squeeze()
xlim = -3e-3, 1e-3
ylim = 1e-3, 5e-3
for img_index, title in [(index, '(b) Streaked'), (where0, '(a) Unstreaked')][::-1]:
    raw_image = sc.plot_list_image[img_index]

    x_axis = sc.plot_list_x[img_index]
    y_axis = sc.y_axis_list[img_index]

    img = iap.Image(raw_image, x_axis, y_axis)
    sp_img = subplot(sp_ctr, title=title, xlabel='x (mm)', ylabel='y (mm)', title_fs=title_fs)
    sp_ctr += 1
    img.plot_img_and_proj(sp_img, xlim=xlim, ylim=ylim, plot_gauss=False)
    sumx = raw_image.sum(axis=0)
    prof = iap.AnyProfile(x_axis, sumx-np.min(sumx))
    prof.cutoff2(3e-2)
    prof.crop()
    prof.reshape(5e3)
    x_rms = prof.rms()
    x_gf = prof.gaussfit.sigma
    distance = sc.gap0/2. - abs(sc.offsets[img_index])
    print('%s RMS: %i um; Gauss sigma: %i um, d=%i um' % (title, round(x_rms*1e6), round(x_gf*1e6), round(distance*1e6)))
    if img_index == where0:
        unstreaked_beamsize = x_gf

sp_profile, sp_screen = [subplot(x+3, grid=False) for x in range(2)]
sp_opt = sp_moments = sp_dummy = lasing.dummy_plot()
sp_ctr += 2

for sp, title, xlabel, ylabel in [
        (sp_profile, '(c) Profile reconstruction', 't (fs)', 'I (kA)'),
        (sp_screen, '(d) Screen reconstruction', 'x (mm)', config.rho_label),
        #(sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
        #(sp_moments, 'Transverse moments', 'Gaussian $\sigma$ (fs)', r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)'),
        ]:
    sp.clear()
    sp.set_title(title, fontsize=title_fs)
    sp.set_xlabel(xlabel)
    sp.set_ylabel(ylabel)

plot_handles = sp_screen, sp_profile, sp_opt, sp_moments

tracker.gauss_prec=1e-15

outp = analysis.current_profile_rec_gauss(tracker, recon_kwargs, do_plot=False)
analysis.plot_rec_gauss(tracker, recon_kwargs, outp, plot_handles, [blmeas_profile], both_zero_crossings=False, skip_indices=(2,))
tracker.gauss_prec=0.5e-15

#sp_screen.get_legend().remove()
#sp_profile.get_legend().remove()


sp_screen_pos = subplot(sp_ctr, title='(e) Distance scan', xlabel='x (mm)', ylabel=config.rho_label)
sp_ctr += 1
sp_profile_pos = subplot(sp_ctr, title='(f) Profile comparison', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1

plot_handles = None, (lasing.dummy_plot(), sp_screen_pos, lasing.dummy_plot(), sp_profile_pos)
beam_offsets, _ = sc.reconstruct_current(tracker, copy.deepcopy(recon_kwargs), force_gap=recon_kwargs['gaps'][1])
sc.plot_reconstruction(plot_handles=plot_handles, blmeas_profile=blmeas_profile, max_distance=300e-6)
sc.plot_reconstruction(plot_handles=None, blmeas_profile=blmeas_profile, max_distance=np.inf)

#sp_screen_pos.get_legend().remove()
#sp_profile_pos.get_legend().remove()




gap = recon_kwargs['gaps'][1]
beam_offset = recon_kwargs['beam_offsets'][-1]
struct_length = 1


gauss_kwargs = config.get_default_gauss_recon_settings()
tracker_kwargs = config.get_default_tracker_settings()
n_emittance = 300e-9
tracker_kwargs['n_emittances'] = [n_emittance, n_emittance]

tracker = tracking.Tracker(**tracker_kwargs)



#blmeas_profile.plot_standard(sp_profile_pos, color='black', ls='--')

#ms.figure('Resolution', figsize=(10, 8))
#ms.plt.subplots_adjust(hspace=0.4, wspace=0.8)
#subplot = ms.subplot_factory(2,3, grid=False)
ms.plt.figure(fig.number)

#image_file = data_dir1+'2021_05_18-21_02_13_Lasing_False_SARBD02-DSCR050.h5'
#image_dict = h5_storage.loadH5Recursive(image_file)
#meta_data1 = image_dict['meta_data_begin']

#images = image_dict['pyscan_result']['image'].astype(float)
#x_axis = image_dict['pyscan_result']['x_axis_m'] - screen_x0
#y_axis = image_dict['pyscan_result']['y_axis_m']
#projx = images.sum(axis=-2)
#median_index = misc.get_median(projx, method='mean', output='index')
#raw_image1 = images[median_index]
#raw_image1 -= np.median(raw_image1)
#image1 = iap.Image(raw_image1, x_axis, y_axis)


#strong_streaking_file = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'
#strong_streaking_dict = h5_storage.loadH5Recursive(strong_streaking_file)
#meta_data2 = strong_streaking_dict['meta_data_begin']
#
#strong_calib_file = data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'
#strong_calib_dict = h5_storage.loadH5Recursive(strong_calib_file)
#screen_x0 = strong_calib_dict['meta_data']['screen_x0']
#index = np.argwhere(strong_calib_dict['meta_data']['offsets'] == 0)
#raw_image = ((strong_calib_dict['raw_data']['pyscan_result']['image'])[index,0]).astype(float).squeeze()
#raw_image2 = ((strong_calib_dict['raw_data']['pyscan_result']['image'])[0,0]).astype(float).squeeze()
#x_axis = strong_calib_dict['raw_data']['pyscan_result']['x_axis_m'] - screen_x0
#y_axis = strong_calib_dict['raw_data']['pyscan_result']['y_axis_m']
#calib_image2 = iap.Image(raw_image, x_axis, y_axis)
#image2 = iap.Image(raw_image2, x_axis, y_axis)



meta_data = sc.meta_data
tracker.set_simulator(meta_data)
blmeas_profile.energy_eV = tracker.energy_eV
tracker.override_quad_beamsize = False
#tracker.n_emittances = [200e-9, 200e-9]

if plot_gap_recon:
    sp_gap = subplot(sp_ctr, title='(g) Gap reconstruction', xlabel='$\Delta$ d ($\mu$m)', ylabel='rms bunch duration (fs)', title_fs=title_fs)
    sp_profile1 = sp_dummy
else:
    sp_gap = subplot(sp_ctr, title='(g) Profile and wake', xlabel='t (fs)', ylabel='Wake (MV/m)', title_fs=title_fs)
    sp_profile1 = sp_gap.twinx()
sp_ctr += 1

sp_res = subplot(sp_ctr, title='(h) Resolution', xlabel='t (fs)', ylabel='R (fs)', title_fs=title_fs)
sp_ctr += 1
sp_profile = sp_res.twinx()


blmeas_profile.plot_standard(sp_profile1, color='black', ls='--')
blmeas_profile.plot_standard(sp_profile, color='black', ls='--', label='I(t)')

sp_profile1.set_yticklabels([])
sp_profile1.set_yticks([])
sp_profile.set_yticks([])

for ctr, (distance, color_ctr) in enumerate([(231e-6, 2), (294e-6, 0)]):
    beam_offset = gap/2. - distance
    wake_dict = blmeas_profile.calc_wake(gap, beam_offset, struct_length)
    wake_t = wake_dict['input']['charge_xx']/c + blmeas_profile.time.min()
    wake_E = wake_dict['dipole']['wake_potential']
    if not plot_gap_recon:
        color = sp_gap.plot(wake_t*1e15, np.abs(wake_E)/1e6, label='%i' % (distance*1e6))[0].get_color()
    else:
        color = ms.plt.rcParams['axes.prop_cycle'].by_key()['color'][color_ctr]

    tracker.n_particles = int(200e3)
    emittances = [tracker.fit_emittance(unstreaked_beamsize, 20e-6, 200e-15), 200e-9]
    emittances = [200e-9]
    print('Emittance X set to %i nm' % (tracker.n_emittances[0]*1e9))
    res_dicts = []
    for emit_ctr, n_emittance in enumerate(emittances):
        for q_ctr, quad_wake in enumerate([True, False]):
            ls = [None, 'dotted'][q_ctr]
            tracker.n_emittances[0] = n_emittance
            tracker.quad_wake = quad_wake
            res_dict = iap.calc_resolution(blmeas_profile, gap, beam_offset, struct_length, tracker, 1)
            res = res_dict['resolution']
            res_t = res_dict['time']

            if q_ctr == 0:
                label = '%i' % (round(distance*1e6))
            else:
                label = None
            mask = res<10e-15
            sp_res.plot(res_t[mask]*1e15, res[mask]*1e15, label=label, color=color, ls=ls)
            res_dicts.append(res_dict)

sp_res.set_ylim(0, 10)
#sp_res.legend(title='d ($\mu$m)', loc='upper right')
ms.comb_legend(sp_res, sp_profile, title='d ($\mu$m)', loc='upper right')

if recon_gap:
    if plot_gap_recon:
        plot_handles = (sp_gap, sp_dummy, sp_dummy, sp_dummy)
    else:
        plot_handles = None
    sc.plot_gap_reconstruction(gap_recon_dict, plot_handles=plot_handles, exclude_gap_ctrs=(2,))
    sc.plot_gap_reconstruction(gap_recon_dict)
    old_lim = sp_gap.get_xlim()
    sp_gap.set_xlim([old_lim[0], old_lim[1]+80])

if not args.noshow:
    ms.show()

if args.save:
    ms.saveall(args.save, hspace, wspace, ending='.pdf')

