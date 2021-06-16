from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
import image_and_profile as iap
import myplotstyle as ms

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
gauss_kwargs = config.get_default_gauss_recon_settings()
tracker = tracking.Tracker(**tracker_kwargs)
n_streaker = 1

streaker_calib_files = (data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5', data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',)
streaker_calib_files = (data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5') # Affected but ok

sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3)
for scf in streaker_calib_files:
    sc.add_file(scf)


sc.fit()
blmeas_profile, _ = sc.forward_propagate(blmeas_file, gauss_kwargs['tt_halfrange'], 200e-12, tracker, blmeas_cutoff=5e-2)

fig, plot_handles = streaker_calibration.streaker_calibration_figure(figsize=(12, 9))

sc.plot_streaker_calib(plot_handles)

offset_list, gauss_dicts = sc.reconstruct_current(tracker, gauss_kwargs)




gap = sc.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]['gap_fit']
streaker_offset = sc.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]['streaker_offset']

beam_offset_arr = -(sc.offsets - streaker_offset)
ms.plt.suptitle('Gaussian reconstruction; gap=%.3f mm' % (gap*1e3))

plot_handles = streaker_calibration.gauss_recon_figure(figsize=(9, 6))
sc.plot_reconstruction(blmeas_profile=blmeas_profile, plot_handles=plot_handles)

ms.figure('Resolution')
subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1

sp_res = subplot(sp_ctr, title='Resolution', xlabel='t (fs)', ylabel='R (fs)')
sp_current = sp_res.twinx()
blmeas_profile.plot_standard(sp_current, color='black')



tracker.n_emittances = [300e-9, 300e-9]
tracker.n_particles = int(200e3)

#blmeas_profile.reshape(int(10e3))

for offset0, beam_offset in zip(sc.offsets, beam_offset_arr):
    if offset0 <= 0:
        continue
    t_axis, resolution = iap.calc_resolution(blmeas_profile, gap, beam_offset, 1, tracker, 1)
    _label = '%.2f mm' % (beam_offset*1e3)
    sp_res.plot(t_axis*1e15, resolution*1e15, label=_label)

sp_res.set_ylim(0, 20)
sp_res.legend()







ms.show()

