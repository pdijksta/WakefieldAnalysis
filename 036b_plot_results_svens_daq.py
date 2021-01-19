import socket
import os
import numpy as np; np
import glob; glob
import matplotlib.pyplot as plt
from EmittanceTool.h5_storage import loadH5Recursive

import myplotstyle as ms


plt.close('all')


fig = ms.figure('BPM trajectories in SARBD02')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1


xlabel = 'Offset [mm]'
ylabel = 'BPM reading [mm]'


sp_x2 = subplot(sp_ctr, title='DEH1 BPM 010 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_x = subplot(sp_ctr, title='DEH1 BPM 040 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_x3 = subplot(sp_ctr, title='DEH2 BPM 010 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_x4 = subplot(sp_ctr, title='DEH2 BPM 040 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1


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

#files_ = sorted(glob.glob('/sf/data/measurements/2020/02/03/Dechirper*.h5'))


for gapnum, gap_file, sps in [
    (1, gap_file1, (sp_x, sp_x2)),
    (2, gap_file2, (sp_x4, sp_x3)),
        ]:

    for gap, files in gap_file:

        if not files:
            continue

        xx_list = []
        bpm1_list = []
        bpm2_list = []
        for file_ in files:
            if socket.gethostname() == 'desktop':
                file_ = os.path.join('/storage/data_2020-02-03', os.path.basename(file_))
            dict_ = loadH5Recursive(file_)

            bpm_data1 = dict_['scan 1']['data']['SARBD02-DBPM040']['X1']
            bpm_data2 = dict_['scan 1']['data']['SARBD02-DBPM010']['X1']


            bpm1_list.append(bpm_data1)
            bpm2_list.append(bpm_data2)

            xx = dict_['scan 1']['method']['actuators']['SARUN18-UDCP%i00' % gapnum]['CENTER']
            xx_list.append(xx)

        len2 = sum(x.shape[-1] for x in bpm1_list)
        bpm1_data = np.zeros((len(xx_list[0]), len2))
        bpm2_data = bpm1_data.copy()

        ctr = 0
        for l, l2 in zip(bpm1_list, bpm2_list):
            ll = l.shape[-1]
            bpm1_data[:, ctr:ctr+ll] = l
            bpm2_data[:, ctr:ctr+ll] = l2
            ctr += ll

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

        sps[0].errorbar(xx_list[0], x1_mean, yerr=x1_err, label=gap)
        sps[1].errorbar(xx_list[0], x2_mean, yerr=x2_err, label=gap)

for sp_ in sp_x, sp_x2, sp_x3, sp_x4:
    sp_.legend()




#0bpm_data = dict_['bpm_data']
#0
#0x_data = bpm_data[0::2]
#0x_mean = np.mean(x_data, axis=-1)
#0x_std = np.std(x_data, axis=-1)
#0
#0
#0y_data = bpm_data[1::2]
#0y_mean = np.mean(y_data, axis=-1)
#0y_std = np.std(y_data, axis=-1)
#0
#0xx = np.arange(len(x_mean))
#0
#0for sps, index in (([sp_x, sp_y], 0), ([sp_x2, sp_y2], -5)):
#0
#0    sps[0].errorbar(xx[index:], x_mean[index:], yerr=x_std[index:])
#0    sps[1].errorbar(xx[index:], y_mean[index:], yerr=y_std[index:])
#0
#0

#ms.saveall('/mnt/usb/work/plots/036b_plot_results_svens_daq')



plt.show()


