import argparse
import numpy as np
from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
import analysis
import image_and_profile as iap
import myplotstyle as ms

parser = argparse.ArgumentParser()
parser.add_argument('file_index', type=int)
parser.add_argument('offset_index', type=int)
parser.add_argument('method', type=str)
parser.add_argument('--noshow', action='store_true')
args = parser.parse_args()

offset_index = args.offset_index
file_index = args.file_index
method = args.method

ms.closeall()
elegant_matrix.set_tmp_dir('~/tmp_elegant/')
ms.set_fontsizes(config.fontsize)

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')

all_streaker_calib = [
        (data_dir+'2021_05_18-17_08_40_Calibration_data_SARUN18-UDCP020.h5',),
        #'2021_05_18-21_58_48_Calibration_data_SARUN18-UDCP020.h5', Bad saved data
        (data_dir+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5',),
        (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5'), # Affected but ok
        #'2021_05_19-00_13_25_Calibration_SARUN18-UDCP020.h5', # Bad data
        #'2021_05_19-00_24_47_Calibration_SARUN18-UDCP020.h5', # Bad data maybe
        (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',),
        ]

blmeas_file = data_dir+'119325494_bunch_length_meas.h5'
tracker_kwargs = config.get_default_tracker_settings()
tracker_kwargs['quad_wake'] = True
tracker_kwargs['profile_cutoff'] = 0.03
tracker_kwargs['bp_smoothen'] = 0
gauss_kwargs = config.get_default_gauss_recon_settings()
tt_halfrange = gauss_kwargs['tt_halfrange']
tracker = tracking.Tracker(**tracker_kwargs)
n_streaker = 1
#offset_index = -1

#streaker_calib_files = (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',)
#streaker_calib_files = (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5')
#streaker_calib_files = (data_dir+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5',)

streaker_calib_files = all_streaker_calib[file_index]

sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3, fit_gap=False, fit_order=False, proj_cutoff=tracker.screen_cutoff)
for scf in streaker_calib_files:
    sc.add_file(scf)

streaker = config.streaker_names[sc.beamline][sc.n_streaker]
self = sc

gap_arr = np.arange(9.87, 10.05001, 0.01)*1e-3
tracker.set_simulator(self.meta_data)
gauss_kwargs['n_streaker'] = self.n_streaker
gauss_kwargs['method'] = method

ms.figure('Gap reconstruction with current reconstruction', figsize=(20, 16))
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile = subplot(sp_ctr, title='Beam profile', xlabel='t (fs)', ylabel='Current (kA)')
sp_ctr += 1

sp_centroid = subplot(sp_ctr, title='Centroid shift', xlabel='$\Delta$d ($\mu$m)', ylabel='Centroid shift (mm)')
sp_ctr += 1

sp_rms = subplot(sp_ctr, title='Beamsize', xlabel='$\Delta$d ($\mu$m)', ylabel='rms (mm)')
sp_ctr += 1

sp_chi = subplot(sp_ctr, title='Measurement - Propagation', xlabel='Gap (mm)', ylabel='Squared deviation')
sp_ctr += 1
sp_opt = sp_chi.twinx()
sp_opt.set_ylabel('Combined opt value (arb. units)')


ny, nx = 2, 2
subplot = ms.subplot_factory(ny, nx)
sp_ctr = np.inf

chi_sq_centroid = np.zeros_like(gap_arr)
chi_sq_rms = np.zeros_like(gap_arr)
opt_values = np.zeros_like(gap_arr)

def chi_squared(meas, expected, weight=1):
    return np.sum(((meas-expected)/weight)**2)

blmeas_profile = iap.profile_from_blmeas(blmeas_file, tt_halfrange, gauss_kwargs['charge'], tracker.energy_eV, True)
blmeas_profile.cutoff2(tracker.profile_cutoff)
blmeas_profile.crop()
blmeas_profile.reshape(tracker.len_screen)

streaker_offset = 376e-6

for gap_ctr, gap in enumerate(gap_arr):

    sc.gap0 = gap

    if sp_ctr > ny*nx:
        fig = ms.figure('Screen distributions', figsize=(20, 16))
        sp_ctr = 1
        fignum = fig.number
    ms.plt.figure(fignum)
    sp_proj = subplot(sp_ctr, title='Screen distributions %.3f' % (gap*1e3), xlabel='x (mm)', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    sc.fit()
    fit_dict_centroid = sc.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]
    streaker_offset2 = fit_dict_centroid['streaker_offset']
    print('Reconstructed offset for gap %.3f: %.3f' % (gap*1e3, streaker_offset2*1e3))

    corrected_distances = gap/2. - np.abs(self.offsets - streaker_offset)
    #offset_index = np.argsort(corrected_distances)[0]

    #fprop0 = sc.forward_propagate(blmeas_file, tt_halfrange, gauss_kwargs['charge'], tracker, blmeas_cutoff=5e-2, force_gap=gap)
    #sc.plot_streaker_calib()
    #ms.plt.suptitle('Gap = %.3f' % (gap*1e3))

    gaps = [10e-3, 10e-3]
    beam_offsets = [0., 0.]

    gaps[self.n_streaker] = gap
    beam_offsets[self.n_streaker] = -(self.offsets[offset_index] - streaker_offset)
    meas_screen = iap.ScreenDistribution(self.plot_list_x[offset_index], self.plot_list_y[offset_index])
    gauss_kwargs['meas_screen'] = meas_screen
    gauss_kwargs['gaps'] = gaps
    gauss_kwargs['beam_offsets'] = beam_offsets
    gauss_kwargs['sig_t_range'] = np.exp(np.linspace(np.log(10), np.log(65), 20))*1e-15

    gauss_dict = analysis.current_profile_rec_gauss(tracker, gauss_kwargs, do_plot=True, figsize=(20, 12))
    ms.plt.suptitle('Gap = %.3f' % (gap*1e3))
    profile = gauss_dict['reconstructed_profile']
    rec_screen = gauss_dict['reconstructed_screen']
    profile.plot_standard(sp_profile, label='%.3f mm; %i fs' % (gap*1e3, profile.rms()*1e15), center='Mean')
    charge = gauss_kwargs['charge']

    #tracker.quad_wake = True
    fprop_dict = self.forward_propagate(profile, tt_halfrange, charge, tracker, force_gap=gap, force_streaker_offset=streaker_offset)
    #tracker.quad_wake = False
    sim_screens = fprop_dict['sim_screens']
    #self.plot_streaker_calib()

    meas_screens = self.get_meas_screens()

    offsets = self.offsets
    rms_sim = np.zeros(len(offsets))
    centroid_sim = rms_sim.copy()

    for n_proj, (_meas_screen, sim_screen, offset) in enumerate(zip(meas_screens, sim_screens, offsets)):
        color = ms.colorprog(n_proj, offsets)
        _meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
        sim_screen.plot_standard(sp_proj, color=color, ls='--')
        centroid_sim[n_proj] = sim_screen.mean()
        rms_sim[n_proj] = sim_screen.rms()
        opt_values[gap_ctr] += _meas_screen.compare(sim_screen)

    nonzero = offsets != 0
    distance_arr = gap/2. - np.abs(self.offsets - streaker_offset)

    xx_plot = distance_arr[nonzero]*1e6
    sort = np.argsort(xx_plot)
    xx_plot = xx_plot[sort]
    xx_plot -= xx_plot.min()
    if gap_ctr == 0:
        sp_rms.errorbar(xx_plot, self.rms[nonzero][sort]*1e3, yerr=self.rms_std[nonzero][sort]*1e3, label=None, ls='None', marker='o', color='black')
        sp_centroid.errorbar(xx_plot, np.abs(self.centroids[nonzero][sort])*1e3, yerr=self.centroids_std[nonzero][sort]*1e3, label=None, ls='None', marker='o', color='black')

    _label = '%.3f' % (gap*1e3)
    sp_rms.plot(xx_plot, rms_sim[nonzero][sort]*1e3, label=_label, marker='.')
    sp_centroid.plot(xx_plot, np.abs(centroid_sim[nonzero][sort])*1e3, label=_label, marker='.')

    chi_sq_rms[gap_ctr] = chi_squared(self.rms[nonzero], rms_sim[nonzero], weight=1) #self.rms_std[nonzero])
    chi_sq_centroid[gap_ctr] = chi_squared(self.centroids[nonzero], centroid_sim[nonzero], weight=1) #self.centroids_std[nonzero])

for yy, label in [(chi_sq_rms, 'Beamsize'), (chi_sq_centroid, 'Centroid')]:
    sp_chi.plot(gap_arr*1e3, yy*1e3, label=label, marker='.')

sp_opt.plot(gap_arr*1e3, opt_values, label='Opt value sum', marker='.', color='red')

ms.comb_legend(sp_chi, sp_opt)

blmeas_profile.plot_standard(sp_profile, color='black', lw=3., center='Mean', label='%i fs' % (blmeas_profile.rms()*1e15))
sp_rms.legend(title='Gap (mm)')
sp_centroid.legend(title='Gap (mm)')
sp_profile.legend()


if gauss_kwargs['method'] in ('rms', 'beamsize'):
    gap = gap_arr[np.argmin(chi_sq_rms)]
elif gauss_kwargs['method'] == 'centroid':
    gap = gap_arr[np.argmin(chi_sq_centroid)]
else:
    gap = gap_arr[np.argmin(opt_values)]
print('Final gap: %.3f mm' % (gap*1e3))

sc.gap0 = gap
sc.fit_gap = False
sc.fit()
sc.plot_streaker_calib(figsize=(20, 12))
ms.plt.suptitle('Final')

offset_list, gauss_dicts = sc.reconstruct_current(tracker, gauss_kwargs)
sc.plot_reconstruction(blmeas_profile=blmeas_profile, figsize=(20, 12))

ms.plt.suptitle('Final')


ms.saveall('./album060_v2/%s_file_%i_offset_%i' % (method, file_index, offset_index), empty_suptitle=False)

if not args.noshow:
    ms.show()

