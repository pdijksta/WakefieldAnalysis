import os
import h5py
from scipy.constants import c
import numpy as np; np
import matplotlib.pyplot as plt

import myplotstyle as ms

import data_loader
import elegant_matrix
import wf_model

#data_dir = '/storage/data_2020-02-03/'
data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
reverse_current_profile = True
linear_fit_details = True
quadratic_fit_details = True

simulator = elegant_matrix.get_simulator(file_json='/afs/psi.ch/intranet//SF/Beamdynamics/Philipp/data/archiver_api_data/2020-02-03.json11')

plt.close('all')

figsize = (16, 12)
ny, nx = 2, 4
subplot = ms.subplot_factory(ny, nx)

xlabel = 'Offset [mm]'
ylabel = 'BPM reading [mm]'

xlabel2 = r's [$\mu$m]'

guessed_centers = [0.46e-3, 0.69e-3]
bpm_reading_0 = [0.3e-3, 0.39e-3]

bl_meas_file = data_dir + 'Bunch_length_meas_2020-02-03_15-59-13.h5'
total_charge = 200e-12

bl_meas = data_loader.load_blmeas(bl_meas_file)
current_profile = bl_meas['current1']

charge_profile = current_profile * total_charge / np.sum(current_profile)
charge_xx = bl_meas['time_profile1']*c
charge_xx -= charge_xx.min()
energy_eV = bl_meas['energy_eV']

wf_calc = wf_model.WakeFieldCalculator(charge_xx, charge_profile, energy_eV)

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


bpms = ['SARUN19-DBPM070', 'SARUN20-DBPM070', 'SARBD02-DBPM010', 'SARBD02-DBPM040']

sp_ctr = np.inf

for n_streaker, gap_file in [
    (1, gap_file1),
    #(2, gap_file2),
        ]:

    for gap_mm, files in gap_file:

        xx_list = []
        bpm_list_dict = {key: [] for key in bpms}
        semigap_m = gap_mm*1e-3/2.
        for n_file, file_ in enumerate(files):
            # Measurement

            file_ = os.path.join(data_dir, os.path.basename(file_))
            _, year, month, day, hour, minute, second, _ = os.path.basename(file_).split('_')
            timestamp = elegant_matrix.get_timestamp(year, month, day, hour, minute, second)
            with h5py.File(file_, 'r') as dict_:
                for bpm in bpms:
                    bpm_list_dict[bpm].append(np.array(dict_['scan 1']['data'][bpm]['X1'])*1e-3)

                offset = np.array(dict_['scan 1']['method']['actuators']['SARUN18-UDCP%i00' % n_streaker]['CENTER'])*1e-3
            xx_list.append(offset)

        # Combine different measurements
        len2 = sum(x.shape[-1] for x in bpm_list_dict[bpms[0]])
        bpm_data = {bpm: np.zeros((len(xx_list[0]), len2)) for bpm in bpms}

        for key, ll in bpm_list_dict.items():
            ctr = 0
            for l_ in ll:
                l_len = l_.shape[-1]
                bpm_data[key][:,ctr:ctr+l_len] = l_
                ctr += l_len

        # filter out non-unique bpm data
        for a in bpm_data.values():
            for n_col in range(a.shape[0]):
                old = a[n_col].copy()
                a[n_col] = np.nan
                arr2 = np.array(list(set(old)))
                a[n_col][:len(arr2)] = arr2

        if sp_ctr > ny*nx:
            sp_ctr = 1
            fig_raw = ms.figure('Streaker gap_mm scan (raw)', figsize=figsize)
            fig_rescaled = ms.figure('Streaker gap_mm scan (normalized by R12)', figsize=figsize)
            fig_wf = ms.figure('Wake functions', figsize=figsize)
            #fig_kick = ms.figure('Kick', figsize=figsize)

        plt.figure(fig_raw.number)
        sp_raw = subplot(sp_ctr, title='Streaker %i Gap %.1f mm' % (n_streaker, gap_mm), xlabel='Offset [mm]', ylabel='BPM reading [mm]')

        plt.figure(fig_rescaled.number)
        sp_rescaled = subplot(sp_ctr, title='Streaker %i Gap %.1f mm' % (n_streaker, gap_mm), xlabel='Offset [mm]', ylabel='BPM reading / R12', sciy=True)

        plt.figure(fig_wf.number)
        sp_trans_wake = subplot(sp_ctr, title='Wakefield %i Gap %.1f mm' % (n_streaker, gap_mm), xlabel='t [s]', ylabel='E [V/m]')

        #plt.figure(fig_kick.number)
        #sp_kick = subplot(sp_ctr, title='Wakefield %i Gap %.1f mm' % (n_streaker, gap_mm), xlabel='t [s]', ylabel='E [V/m]')
        sp_ctr += 1

        xx_plot = (xx_list[0] - guessed_centers[n_streaker-1])*1e3
        xx_plot_m = xx_plot*1e-3
        beam_offset_list = np.linspace(xx_plot_m.min(), xx_plot_m.max(), 10)

        simulated_kick_normalized = []

        for beam_offset in beam_offset_list:

            if n_streaker == 1:
                gaps = (gap_mm*1e-3, 20e-3)
                beam_offsets = (beam_offset, 0)
            else:
                gaps = (20e-3, gap_mm*1e-3)
                beam_offsets = (0, beam_offset)
            sim, mat_dict, wf_dicts, _ = simulator.simulate_streaker(charge_xx/c, current_profile, timestamp, gaps, beam_offsets, energy_eV, linearize_twf=False)
            bpm_index = list(sim.cen['ElementName']).index('SARBD02.DBPM040')
            simulated_kick_normalized.append(sim.cen['Cx'][bpm_index])
            #break
        simulated_kick_normalized = np.array(simulated_kick_normalized)

        for n_bpm, (bpm, arr) in enumerate(bpm_data.items()):

            #mat_dict = simulator.get_elegant_matrix(n_streaker-1, timestamp, print_=True)
            #mat_old = mat_dict[bpm.replace('-','.')]

            mat0 = mat_dict['MIDDLE_STREAKER_%i' % n_streaker]
            mat1 = mat_dict[bpm.replace('-', '.')]
            mat = mat1 @ np.linalg.inv(mat0)
            r12 = mat[0,1]

            x1_mean = np.nanmean(arr, axis=-1)
            x1_err = np.nanstd(arr, axis=-1)

            beam_offset_model = (np.linspace(xx_plot.min(), xx_plot.max(), 100)*1e-3)[:, np.newaxis]
            wf_dict = wf_calc.calc_all(semigap_m, r12, beam_offset=beam_offset_model, calc_lin_dipole=False, calc_quadrupole=False, calc_long_dipole=False)
            #plt.figure()
            #plt.plot(wf_dicts[0]['t']*c, -wf_dicts[0]['WX'], label='WX')
            #plt.plot(wf_dict['input']['charge_xx'], wf_dict['dipole']['single_particle_wake'][0,:], label='wld')
            #plt.legend()
            #plt.show()
            #import pdb; pdb.set_trace()
            print('%.1e %.2e' % (semigap_m*2*1e3, wf_dict['dipole']['single_particle_wake'].max()))
            kick = wf_dict['dipole']['kick']
            kick_effect = wf_dict['dipole']['kick_effect']

            index0 = np.argmin(xx_plot**2)
            guessed0 = np.mean(x1_mean[index0-1:index0+1])
            color = sp_raw.errorbar(xx_plot, (x1_mean-guessed0)*1e3, yerr=x1_err*1e3, label=bpm).lines[0].get_color()
            sp_raw.plot(beam_offset_model*1e3, -kick_effect*1e3, ls='--', color=color)

            sp_rescaled.errorbar(xx_plot, (x1_mean-guessed0)/r12, yerr=x1_err/r12, label=bpm)

            if n_bpm == 0:
                sp_rescaled.plot(beam_offset_model*1e3, -kick, ls='--', color='black', label='Model')

                index = np.argmin((wf_dict['input']['beam_offset'].squeeze() - beam_offset)**2)
                spw1 = -wf_dict['dipole']['single_particle_wake'][index]
                tt1 = wf_dict['input']['charge_xx']/c
                spw2 = wf_dicts[n_streaker-1]['WX']
                tt2 = wf_dicts[n_streaker-1]['t']

                sp_trans_wake.plot(tt1, spw1, label='Model')
                sp_trans_wake.plot(tt2, spw2, label='Elegant')

            if bpm == 'SARBD02-DBPM040':
                sp_rescaled.plot((beam_offset_list)*1e3, simulated_kick_normalized/r12, ls='-.', color='black', label='Elegant', marker='.')

        for sp in sp_raw, sp_rescaled:
            sp.legend(title='BPM')

ms.saveall('/tmp/005b_include_other', ending='.png', bottom=0.15, wspace=0.3)
del sim

plt.show()

