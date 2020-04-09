import os
import datetime
import copy
import itertools
import h5py
import numpy as np; np
from scipy.constants import c
import scipy.optimize as opt
import matplotlib.pyplot as plt

#from EmittanceTool.h5_storage import loadH5Recursive

import myplotstyle as ms

import wf_model
import data_loader
import elegant_matrix

data_dir = '/storage/data_2020-02-03/'
bpms = ['SARBD02.DBPM010', 'SARBD02.DBPM040']

reverse_current_profile = True
linear_fit_details = True
quadratic_fit_details = True


plt.close('all')


figsize = (16, 12)
fig = ms.figure('BPM trajectories in SARBD02', figsize=figsize)
subplot = ms.subplot_factory(2,3)
sp_ctr = 1


xlabel = 'Offset [mm]'
ylabel = 'BPM reading [mm]'


sp_x = subplot(sp_ctr, title='DEH1 BPM 010 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_x2 = subplot(sp_ctr, title='DEH1 BPM 040 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1


sp_x3 = subplot(sp_ctr, title='DEH2 BPM 010 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

sp_x4 = subplot(sp_ctr, title='DEH2 BPM 040 X', xlabel=xlabel, ylabel=ylabel)
sp_ctr += 1

xlabel2 = 's [$\mu$m]'
sp_current = subplot(sp_ctr, title='Current profile', xlabel=xlabel2, ylabel='Current profile [kA]')
sp_ctr += 1

sp_wake = subplot(sp_ctr, title='Wake potential', xlabel=xlabel2, ylabel='E [MV/m m(offset)]')
sp_ctr += 1

guessed_centers = [0.46e-3, 0.70e-3]
bpm_reading_0 = [0.3e-3, 0.4e-3]

bl_meas_file = data_dir + 'Bunch_length_meas_2020-02-03_15-59-13.h5'
total_charge = 200e-12

bl_meas = data_loader.load_blmeas(bl_meas_file)
current_profile = bl_meas['current1']

if reverse_current_profile:
    current_profile = current_profile[::-1]

charge_profile = current_profile * total_charge / np.sum(current_profile)
charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
energy_eV = bl_meas['energy_eV']

wf_calc = wf_model.WakeFieldCalculator(charge_xx, charge_profile)

sp_current.plot(charge_xx*1e6, current_profile/1e3)


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

plotted_gaps = []

exp_results = {
        1: {1: {}, 2: {}},
        2: {1: {}, 2: {}},
        }

model_results = copy.deepcopy(exp_results)


for n_streaker, gap_file, sps in [
    (1, gap_file1, (sp_x, sp_x2)),
    (2, gap_file2, (sp_x3, sp_x4)),
        ]:

    for gap, files in gap_file:

        semigap_m = gap/2 * 1e-3
        spw = wf_calc.get_single_particle_wake(semigap_m)
        wake_potential = wf_calc.get_wake_potential(spw)
        #wf_dict = wf_calc.calc_all(semigap_m, 10, energy_eV)
        #wake_potential = wf_dict['wake_potential']

        if gap not in plotted_gaps:
            sp_wake.plot(charge_xx*1e6, wake_potential*1e-6, label='%.1f' % gap)
            plotted_gaps.append(gap)

        xx_list = []
        bpm1_list, bpm2_list = [], []
        bpm1_list_model, bpm2_list_model = [], []

        for n_file, file_ in enumerate(files):
            # Model
            if n_file == 0:
                _, year, month, day, hour, minute, second, _ = os.path.basename(file_).split('_')
                date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                timestamp = int(date.strftime('%s'))
                mat = elegant_matrix.get_elegant_matrix(n_streaker-1, timestamp)
                r12_bpm1 = mat[bpms[0]][0,1]
                r12_bpm2 = mat[bpms[1]][0,1]

                wf_dict_bpm1 = wf_calc.calc_all(semigap_m, r12_bpm1, energy_eV)
                wf_dict_bpm2 = wf_calc.calc_all(semigap_m, r12_bpm2, energy_eV)
                bpm1_list_model.append(wf_dict_bpm1['x_per_m_offset'])
                bpm2_list_model.append(wf_dict_bpm2['x_per_m_offset'])

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

        xx_plot = (xx_list[0] - guessed_centers[n_streaker-1])*1e3
        line = sps[0].errorbar(xx_plot, (x1_mean-bpm_reading_0[0])*1e3, yerr=x1_err*1e3, label=gap).lines[0]
        sps[1].errorbar(xx_plot, (x2_mean-bpm_reading_0[1])*1e3, yerr=x2_err*1e3, label=gap)

        exp_results[n_streaker][1][gap] = (xx_plot*1e-3, x1_mean, x1_err)
        exp_results[n_streaker][2][gap] = (xx_plot*1e-3, x2_mean, x2_err)

        # Theory
        model_kick1 = bpm1_list_model[0]*xx_plot*1e-3*(-1)
        model_kick2 = bpm2_list_model[0]*xx_plot*1e-3*(-1)
        sps[0].plot(xx_plot, model_kick1*1e3, color=line.get_color(), ls='--')
        sps[1].plot(xx_plot, model_kick2*1e3, color=line.get_color(), ls='--')

        model_results[n_streaker][1][gap] = bpm1_list_model[0]
        model_results[n_streaker][2][gap] = bpm2_list_model[0]


for sp_ in sp_x, sp_x2, sp_x3, sp_x4, sp_wake:
    sp_.legend()


ms.figure('Details', figsize=figsize)

sp_ctr = 1
#sp_123 = subplot(sp_ctr, xlabel=xlabel, ylabel=ylabel, title='Linear fit to selected data')
#sp_ctr += 1


sp_scaling = subplot(sp_ctr, xlabel='Gap [mm]', ylabel='Linear offset', title='Scaling with gap size and 4th order fit')
sp_ctr += 1

if linear_fit_details:
    sp_ctr_details = 100
    fig_ctr = 1


ratios = []
fit_results = copy.deepcopy(model_results)
for k1, k2, k3 in itertools.product([1,2], [1,2], [2.5, 3, 4, 6]):
    exp = exp_results[k1][k2][k3]
    model = model_results[k1][k2][k3]
    xx, yy, err = exp
    xx = xx
    yy = yy

    lim = {2.5: 0.2, 3: 0.4, 4: 0.5, 6: 1}[k3]*1e-3
    data_window = np.logical_and(-lim < xx, xx < lim)
    xx_fit = xx[data_window]
    yy_fit = yy[data_window]

    fit = np.poly1d(np.polyfit(xx_fit, yy_fit, 1))
    fit_results[k1][k2][k3] = fit[1]

    ratio = abs(fit[1]/model)
    ratios.append(ratio)

    if linear_fit_details:
        if sp_ctr_details > 2*3:
            fig_details = ms.figure('Linear fit details %i' % fig_ctr, figsize=figsize)
            fig_ctr += 1
            sp_ctr_details = 1
        sp_123 = subplot(sp_ctr_details, title='Device %i bpm %i gap %.1f' % (k1, k2, k3), xlabel='Beam offset [mm]', ylabel='BPM reading [mm]')
        sp_ctr_details += 1

        sp_123.errorbar(xx*1e3, (yy-fit[0])*1e3, yerr=err*1e3, label='Data')
        sp_123.plot(xx_fit*1e3, (fit(xx_fit)-fit[0])*1e3, ls='--', label='Linear fit')
        sp_123.plot(xx_fit*1e3, xx_fit*(-1)*abs(model)*1e3, label='Linear model')
        sp_123.legend()

ratios = np.array(ratios)

if quadratic_fit_details:
    fudge_factor = 4.
    fig = ms.figure('Quadratic fit details (using fudge factor %.1f for model data)', figsize=figsize)
    sp_ctr_quadratic = 1

def order_fit(xx, scale, order):
    return scale/xx**order

def second_order_fit(xx, scale):
    return order_fit(xx, scale, 2)

ratios_f4 = []
order_fits_model = []
order_fits_fit = []

for k1, k2 in itertools.product([1, 2], repeat=2):
    fit_scaling = np.abs(np.array(sorted(fit_results[k1][k2].items())))
    model_scaling = np.abs(np.array(sorted(model_results[k1][k2].items())))

    fit_gap, fit_lin = fit_scaling[:,0]*1e-3, fit_scaling[:,1]
    model_gap, model_lin = model_scaling[:,0]*1e-3, model_scaling[:,1]

    f2_fit = opt.curve_fit(second_order_fit, fit_gap, fit_lin)[0]
    f2_model = opt.curve_fit(second_order_fit, model_gap, model_lin)[0]

    fg_fit = opt.curve_fit(order_fit, fit_gap, fit_lin)[0]
    fg_model = opt.curve_fit(order_fit, model_gap, model_lin)[0]

    ratios_f4.append(abs(f2_fit/f2_model))
    order_fits_model.append(f2_model)
    order_fits_fit.append(f2_fit)

    fit_xx = np.linspace(fit_gap.min(), fit_gap.max(), 100)
    label = 'Device %i BPM %i' % (k1, k2)
    line = sp_scaling.plot(fit_gap*1e3, fit_lin, marker='.', label=label)[0]
    sp_scaling.plot(model_gap*1e3, model_lin, marker='.', color=line.get_color(), ls='--')
    #sp_scaling.plot(fit_xx*1e3, second_order_fit(fit_xx, *f2_fit), ls='-.', color=line.get_color())

    if quadratic_fit_details:
        sp_11 = subplot(sp_ctr_quadratic, xlabel='Gap [mm]', ylabel='Linear scaling (dx / offset)', title='2nd order fit to device %i bpm %i' % (k1, k2))
        sp_ctr_quadratic += 1
        sp_11.plot(fit_gap*1e3, fit_lin, marker='.', label='Linear scaling with beam offset (exp)')
        sp_11.plot(model_gap*1e3, fudge_factor*model_lin, marker='.', label='Linear scaling with beam offset (model)')
        sp_11.plot(fit_xx*1e3, second_order_fit(fit_xx, *f2_fit), ls='--', label='Quadratic fit to scaling')
        sp_11.plot(fit_xx*1e3, order_fit(fit_xx, *fg_fit), ls='--', label='1/g^n fit to scaling with n=%.1f' % fg_fit[1])
        sp_11.plot(fit_xx*1e3, fudge_factor*order_fit(fit_xx, *fg_model), ls='--', label='1/g^n fit to model scaling with n=%.1f' % fg_model[1])
        sp_11.legend()

sp_scaling.legend()

ms.saveall('~/Dropbox/plots/003_trans_wake_first_order', ending='.pdf', bottom=0.15, wspace=0.3)

plt.show()

