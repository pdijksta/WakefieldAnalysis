import matplotlib.pyplot as plt

import h5_storage
import analysis
import myplotstyle as ms

plt.close('all')

default_dir = '/sf/data/measurements/2021/04/25/'
streaker_calib_files = [
        default_dir + '2021_04_25-16_55_25_Calibration_SARUN18-UDCP020.h5',
        ]

ms.figure('Compare streaker calibrations')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_center = subplot(sp_ctr, title='Centroid shift', xlabel='Streaker center (mm)', ylabel='Beam X centroid (mm)')
sp_ctr += 1

sp_sizes = subplot(sp_ctr, title='Size increase', xlabel='Streaker center (mm)', ylabel='Beam X rms (mm)')
sp_ctr += 1

for ctr, calib_file in enumerate(streaker_calib_files):
    dict0 = h5_storage.loadH5Recursive(calib_file)['raw_data']
    dict_ = analysis.analyze_streaker_calibration(dict0)['meta_data']
    plt.suptitle(calib_file)
    streaker_offset = dict_['streaker_offset']
    screen_x0 = dict_['screen_x0']
    offsets = dict_['offsets']

    xx_plot = (offsets-streaker_offset)*1e3
    yy_plot = dict_['centroid_mean'] - screen_x0
    color = sp_center.errorbar(xx_plot, yy_plot*1e3, yerr=dict_['centroid_std']*1e3, label=ctr, ls='None', lw=3)[0].get_color()
    sp_center.plot((dict_['fit_xx']-streaker_offset)*1e3, (dict_['fit_reconstruction']-screen_x0)*1e3, color=color, ls='--')

    yy_plot = dict_['rms_mean'] - screen_x0
    sp_sizes.errorbar(xx_plot, yy_plot*1e3, yerr=dict_['rms_std']*1e3, label=ctr, ls='None')


sp_center.legend()
sp_sizes.legend()


plt.show()

