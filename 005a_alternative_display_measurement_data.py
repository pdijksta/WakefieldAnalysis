import os
import h5py
import numpy as np; np
import matplotlib.pyplot as plt

#from EmittanceTool.h5_storage import loadH5Recursive

import myplotstyle as ms

#import wf_model
#import data_loader
#import elegant_matrix

data_dir = '/storage/data_2020-02-03/'
bpms = ['SARBD02.DBPM010', 'SARBD02.DBPM040']

reverse_current_profile = True
linear_fit_details = True
quadratic_fit_details = True


plt.close('all')


figsize = (16, 12)
ny, nx = 2, 4
subplot = ms.subplot_factory(ny, nx)


xlabel = 'Offset [mm]'
ylabel = 'BPM reading [mm]'

xlabel2 = 's [$\mu$m]'

guessed_centers = [0.46e-3, 0.70e-3]
bpm_reading_0 = [0.3e-3, 0.4e-3]


gap_file1 = sorted([
        (2.5, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_06_45_640410.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_08_55_023858.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_10_37_130346.h5',
            ]),
        (3, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_34_17_349626.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_32_56_278097.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_28_45_243502.h5',
            ]),
        (4, [
            #'/sf/data/measurements/2020/02/03/Dechirper Gap Scan_2020_02_03_17_15_01_320680.h5', wrong shape of data
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_44_05_547434.h5',
            '/sf/data/measurements/2020/02/03/Dechirper Gap Scan_2020_02_03_17_20_35_050005.h5',
            '/sf/data/measurements/2020/02/03/Dechirper Gap Scan_2020_02_03_17_21_28_005167.h5',
            ]),
        (6, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_52_27_374485.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_54_12_164180.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_17_55_24_081062.h5',
             ]),

        ])

gap_file2 = sorted([
        (6, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_26_33_182965.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_28_24_210768.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_29_40_217695.h5',
            ]),
        (4, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_36_58_084502.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_39_42_204489.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_41_16_808553.h5',
            ]),
        (3, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_46_43_066571.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_47_55_366962.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_49_11_173327.h5',
            ]),
        (2.5, [
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_57_30_692370.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_56_16_144374.h5',
            '/sf/data/measurements/2020/02/03/DechirperGapScan_2020_02_03_18_54_54_508553.h5',
            ]),
        ])


sp_ctr = np.inf

for n_streaker, gap_file in [
    (1, gap_file1),
    (2, gap_file2),
        ]:

    for gap, files in gap_file:

        xx_list = []
        bpm1_list, bpm2_list = [], []
        bpm1_list_model, bpm2_list_model = [], []

        for n_file, file_ in enumerate(files):
            # Measurement

            #file_ = os.path.join('/sf/data/measurements/2020/02/03', file_)
            file_ = os.path.join(data_dir, os.path.basename(file_))
            with h5py.File(file_, 'r') as dict_:
                #dict_ = loadH5Recursive(file_)

                bpm_data1 = np.array(dict_['scan 1']['data']['SARBD02-DBPM040']['X1'])*1e-3
                bpm_data2 = np.array(dict_['scan 1']['data']['SARBD02-DBPM010']['X1'])*1e-3

                bpm1_list.append(bpm_data1)
                bpm2_list.append(bpm_data2)

                offset = np.array(dict_['scan 1']['method']['actuators']['SARUN18-UDCP%i00' % n_streaker]['CENTER'])*1e-3
            xx_list.append(offset)

        # Combine different measurements
        len2 = sum(x.shape[-1] for x in bpm1_list)
        bpm1_data = np.zeros((len(xx_list[0]), len2))
        bpm2_data = bpm1_data.copy()

        ctr = 0
        for l, l2 in zip(bpm1_list, bpm2_list):
            ll = l.shape[-1]
            bpm1_data[:,ctr:ctr+ll] = l
            bpm2_data[:,ctr:ctr+ll] = l2
            ctr += ll

        # filter out non-unique bpm data
        for a in bpm1_data, bpm2_data:
            for n_col in range(a.shape[0]):
                old = a[n_col].copy()
                a[n_col] = np.nan
                arr2 = np.array(list(set(old)))
                a[n_col][:len(arr2)] = arr2

        x1_mean = np.nanmean(bpm1_data, axis=-1)
        x1_err = np.nanstd(bpm1_data, axis=-1)

        x2_mean = np.nanmean(bpm2_data, axis=-1)
        x2_err = np.nanstd(bpm2_data, axis=-1)

        if sp_ctr > ny*nx:
            sp_ctr = 1
            fig = ms.figure('Streaker gap scan', figsize=figsize)

        sp = subplot(sp_ctr, title='Streaker %i Gap %.1f mm' % (n_streaker, gap), xlabel='Offset [mm]', ylabel='BPM reading')
        sp_ctr += 1

        xx_plot = (xx_list[0] - guessed_centers[n_streaker-1])*1e3
        sp.errorbar(xx_plot, (x1_mean-bpm_reading_0[0])*1e3, yerr=x1_err*1e3, label='010')
        sp.errorbar(xx_plot, (x2_mean-bpm_reading_0[1])*1e3, yerr=x2_err*1e3, label='040')

        sp.legend(title='BPM')




ms.saveall('~/Dropbox/plots/005a_alternate', ending='.pdf', bottom=0.15, wspace=0.3)

plt.show()

