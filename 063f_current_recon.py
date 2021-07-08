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
import gaussfit
import streaker_calibration
import image_and_profile as iap
import myplotstyle as ms
import elegant_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--save', type=str)
args = parser.parse_args()

ms.closeall()

title_fs = 10

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
recon_gap = False

#sc_dict = h5_storage.loadH5Recursive(sc_file)
#sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3, sc_dict)



sc = streaker_calibration.StreakerCalibration('Aramis', 1, 10e-3)
for scf in (data_dir1+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'):
    sc.add_file(scf)

sc.fit_type('centroid')


tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
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
    gap_arr = np.array([10e-3-100e-6, 10e-3+50e-6])
    gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, recon_kwargs, streaker_offset)

    delta_gap = gap_recon_dict['gap'] - 10e-3
else:
    delta_gap = -55e-6
print('Delta gap %i um' % (delta_gap*1e6))


tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(sc.meta_data)

recon_kwargs['gaps'] = [10e-3, 10e-3+delta_gap]
recon_kwargs['beam_offsets'] = [0., -(sc.offsets[index] - streaker_offset)]
recon_kwargs['n_streaker'] = 1
recon_kwargs['meas_screen'] = meas_screen


hspace, wspace = 0.5, 0.4
fig = ms.figure('Current profile reconstruction', figsize=(6, 12))
ms.plt.subplots_adjust(hspace=hspace, wspace=wspace)
subplot = ms.subplot_factory(4, 2, grid=False)
sp_ctr = 1


where0 = np.argwhere(sc.offsets == 0).squeeze()
xlim = -3e-3, 1e-3
ylim = 1e-3, 5e-3
for img_index, title in [(index, '(b) Streaked'), (where0, '(a) Unstreaked')][::-1]:
    raw_image = sc.images[img_index][0]

    x_axis = sc.plot_list_x[img_index]
    y_axis = sc.y_axis_list[img_index]

    img = iap.Image(raw_image, x_axis, y_axis)
    sp_img = subplot(sp_ctr, title=title, xlabel='x (mm)', ylabel='y (mm)', title_fs=title_fs)
    sp_ctr += 1
    img.plot_img_and_proj(sp_img, xlim=xlim, ylim=ylim, plot_gauss=False)
    sumx = raw_image.sum(axis=0)
    prof = iap.AnyProfile(x_axis, sumx-np.min(sumx))
    prof.cutoff2(0.05)
    prof.crop()
    prof.reshape(1000)
    x_rms = prof.rms()
    x_gf = prof.gaussfit.sigma
    print('%s RMS: %i um; Gauss sigma: %i um' % (title, x_rms*1e6, x_gf*1e6))

sp_profile, sp_screen = [subplot(x+3, grid=False) for x in range(2)]
sp_opt = sp_moments = lasing.dummy_plot()
sp_ctr += 2

for sp, title, xlabel, ylabel in [
        (sp_screen, '(d) Screen rec', 'x (mm)', 'Intensity (arb. units)'),
        (sp_profile, '(c) Profile rec', 't (fs)', 'Current (kA)'),
        (sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
        (sp_moments, 'Transverse moments', 'Gaussian $\sigma$ (fs)', r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)'),
        ]:
    sp.clear()
    sp.set_title(title, fontsize=title_fs)
    sp.set_xlabel(xlabel)
    sp.set_ylabel(ylabel)

plot_handles = sp_screen, sp_profile, sp_opt, sp_moments

outp = analysis.current_profile_rec_gauss(tracker, recon_kwargs, plot_handles, blmeas_file, both_zero_crossings=False)

sp_screen.get_legend().remove()
sp_profile.get_legend().remove()


sp_profile_pos = subplot(sp_ctr, title='(e) Profiles', xlabel='t (fs)', ylabel='I (kA)')
sp_ctr += 1
sp_screen_pos = subplot(sp_ctr, title='(f) Screens', xlabel='x (mm)', ylabel='Intensity (arb. units)')
sp_ctr += 1

plot_handles = None, (lasing.dummy_plot(), sp_screen_pos, lasing.dummy_plot(), sp_profile_pos)
beam_offsets, _ = sc.reconstruct_current(tracker, copy.deepcopy(recon_kwargs), force_gap=recon_kwargs['gaps'][1])
sc.plot_reconstruction(plot_handles=plot_handles)

sp_screen_pos.get_legend().remove()
sp_profile_pos.get_legend().remove()




gap = recon_kwargs['gaps'][1]
beam_offset = recon_kwargs['beam_offsets'][-1]
struct_length = 1

gauss_kwargs = config.get_default_gauss_recon_settings()
tracker_kwargs = config.get_default_tracker_settings()

n_emittance = 300e-9
tracker_kwargs['n_emittances'] = [n_emittance, n_emittance]

tracker = tracking.Tracker(**tracker_kwargs)


blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'
blmeas_profile = iap.profile_from_blmeas(blmeas_file, gauss_kwargs['tt_halfrange'], gauss_kwargs['charge'], 0, True)
blmeas_profile.cutoff2(0.03)
blmeas_profile.crop()
blmeas_profile.reshape(1000)

blmeas_profile.plot_standard(sp_profile_pos, color='black')

#ms.figure('Resolution', figsize=(10, 8))
#ms.plt.subplots_adjust(hspace=0.4, wspace=0.8)
#subplot = ms.subplot_factory(2,3, grid=False)
ms.plt.figure(fig.number)

#image_file = data_dir1+'2021_05_18-21_02_13_Lasing_False_SARBD02-DSCR050.h5'
#image_dict = h5_storage.loadH5Recursive(image_file)
#meta_data1 = image_dict['meta_data_begin']

screen_calib_file = data_dir1+'2021_05_18-16_39_27_Screen_Calibration_SARBD02-DSCR050.h5'
screen_calib_dict = h5_storage.loadH5Recursive(screen_calib_file)

screen_calib_raw_image = screen_calib_dict['pyscan_result']['image'][0].astype(float)
x_axis_calib = screen_calib_dict['pyscan_result']['x_axis_m']
screen_x0 = gaussfit.GaussFit(x_axis_calib, screen_calib_raw_image.sum(axis=0)).mean
x_axis_calib -= screen_x0
y_axis_calib = screen_calib_dict['pyscan_result']['y_axis_m']
screen_calib_raw_image -= np.median(screen_calib_raw_image)
screen_calib_image = iap.Image(screen_calib_raw_image, x_axis_calib, y_axis_calib)

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
calib_image = screen_calib_image
tracker.set_simulator(meta_data)
blmeas_profile.energy_eV = tracker.energy_eV
tracker.override_quad_beamsize = False
tracker.n_emittances = [200e-9, 200e-9]

sp_wake = subplot(sp_ctr, title='(g) Profile and wake', xlabel='t (fs)', ylabel='Wake (MV/m)', title_fs=title_fs)
sp_ctr += 1
sp_profile1 = sp_wake.twinx()
#sp_wake.set_ylabel('Wake (MV/m)')
sp_profile = subplot(sp_ctr, title='(h) Resolution', xlabel='t (fs)', ylabel='I (kA)', title_fs=title_fs)
sp_ctr += 1
sp_res = sp_profile.twinx()
sp_res.set_ylabel('R (fs)')


blmeas_profile.plot_standard(sp_profile1, color='black', ls='--')
blmeas_profile.plot_standard(sp_profile, color='black', ls='--')

sp_profile1.set_yticklabels([])

for ctr, beam_offset in enumerate(beam_offsets[-4:][::-1]):
    d = gap/2. - abs(beam_offset)
    wake_dict = blmeas_profile.calc_wake(gap, beam_offset, struct_length)
    wake_t = wake_dict['input']['charge_xx']/c + blmeas_profile.time.min()
    wake_E = wake_dict['dipole']['wake_potential']
    color = sp_wake.plot(wake_t*1e15, np.abs(wake_E)/1e6, label='%i' % (d*1e6))[0].get_color()


    tracker.n_particles = int(200e3)
    res_dicts = []
    if beam_offset in (beam_offsets[-4], beam_offsets[-1]):
        for quad_wake, label, ls in [(False, 'D', '--'), (True, 'Q', None)]:
            tracker.quad_wake = quad_wake
            res_dict = iap.calc_resolution(blmeas_profile, gap, beam_offset, struct_length, tracker, 1)
            res = res_dict['resolution']
            res_t = res_dict['time']

            sp_res.plot(res_t*1e15, res*1e15, label=label, color=color, ls=ls)
            res_dicts.append(res_dict)

    #sp_image = subplot(sp_ctr, title='Raw image', xlabel='x (mm)', ylabel='y (mm)', grid=False)
    #sp_ctr += 1
    #image.plot_img_and_proj(sp_image)

    #gfX = gaussfit.GaussFit(x_axis_calib, screen_calib_raw_image.sum(axis=0))
    #beamsize = gfX.sigma

    #sp_screen_calib = subplot(sp_ctr, title='Screen calibration', xlabel='x (mm)', ylabel='y (mm)', grid=False)
    #sp_ctr += 1
    #calib_image.plot_img_and_proj(sp_screen_calib)

    #resolution2 = beamsize / res_dicts[0]['streaking_strength']
    #time2 = res_dicts[0]['time']

#sp_res.plot(time2*1e15, resolution2*1e15, label='D ? nm')


sp_res.set_ylim(0, 10)

sp_wake.legend(title='d ($\mu$m)', framealpha=1)
sp_wake.set_xlim(-80, None)


#sp_res.legend()

if recon_gap:
    sc.plot_gap_reconstruction(gap_recon_dict, streaker_offset)

if not args.noshow:
    ms.show()

if args.save:
    ms.saveall(args.save, hspace, wspace, ending='.pdf')

