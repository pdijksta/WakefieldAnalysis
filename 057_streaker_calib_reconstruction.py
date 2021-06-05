from socket import gethostname

import elegant_matrix
import streaker_calibration
import config
import tracking
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

for fit_gap in [True, False]:
    sc.fit_gap = fit_gap

    sc.fit()
    blmeas_profile, _ = sc.forward_propagate(blmeas_file, gauss_kwargs['tt_halfrange'], 200e-12, tracker, blmeas_cutoff=5e-2)
    sc.plot_streaker_calib()


    offset_list, gauss_dicts = sc.reconstruct_current(tracker, gauss_kwargs)

    gap = sc.fit_dicts_gap_order['centroid'][sc.fit_gap][sc.fit_order]['gap_fit']

    ms.figure('Gaussian reconstruction; gap=%.3f mm' % (gap*1e3))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1
    sp_screen_pos = subplot(sp_ctr, title='Measured screens pos', xlabel='x (mm)', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_screen_neg = subplot(sp_ctr, title='Measured screens neg', xlabel='x (mm)', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_profile_pos = subplot(sp_ctr, title='Reconstructed profiles pos', xlabel='t (fs)', ylabel='Current (kA)')
    sp_ctr += 1
    sp_profile_neg = subplot(sp_ctr, title='Reconstructed profiles neg', xlabel='t (fs)', ylabel='Current (kA)')
    sp_ctr += 1

    center = 'Mean'

    for _sp in sp_profile_pos, sp_profile_neg:
        blmeas_profile.plot_standard(_sp, color='black', label='TDC', lw=3, center=center)

    for offset, gauss_dict in zip(offset_list, gauss_dicts):
        beam_offset = gauss_dict['beam_offsets'][n_streaker]

        sp_screen = sp_screen_pos if beam_offset > 0 else sp_screen_neg
        sp_profile = sp_profile_pos if beam_offset > 0 else sp_profile_neg

        semigap = gauss_dict['gaps'][n_streaker]/2
        distance = semigap-abs(beam_offset)
        if distance > 350e-6: continue

        label = '%i $\mu$m' % round(distance*1e6)
        rec_profile = gauss_dict['reconstructed_profile']
        rec_profile.plot_standard(sp_profile, label=label, center=center)

        meas_screen = gauss_dict['meas_screen']
        rec_screen = gauss_dict['reconstructed_screen']
        color = meas_screen.plot_standard(sp_screen, label=label)[0].get_color()
        rec_screen.plot_standard(sp_screen, ls='--', color=color)

    for _sp in sp_screen_pos, sp_screen_neg, sp_profile_pos, sp_profile_neg:
        _sp.legend()


ms.show()

