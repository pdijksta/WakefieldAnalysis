import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.io import loadmat
import uwf_model as uwf
import matplotlib.pyplot as plt
from EmittanceTool.h5_storage import loadH5Recursive

import myplotstyle as ms

plt.close('all')

data_dir = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03/'
meas1, meas2, meas3 = 0, 0, 0
mat_files_current_charge = [
        ('Eloss_UNDbis.mat', meas1, 200),
        ('Eloss_UND2ndStreak.mat', meas1, 200),
        ('Eloss_UND-COMP2.mat', meas2, 100),
        ('Eloss_UND-COMP3.mat', meas3, 100),

        #('Eloss_DEH1-COMP1.mat', meas1, 200),
        #('Eloss_DEH1-COMP2.mat', meas2, 200),
        #('Eloss_DEH1-COMP3.mat', meas3, 100),
       ]
L = 58.8
L0 = 1.
L_factor = L/L0

for mat_file, _, total_charge in mat_files_current_charge:
    result_dict = loadH5Recursive(os.path.basename(mat_file)+'_wake.h5')
    dd = loadmat(data_dir+mat_file)
    charge_profile = result_dict['charge_profile']
    charge_xx = result_dict['charge_xx']
    energy_eV = result_dict['energy_eV']
    gap_list = result_dict['gap_list']
    #result_dict = {str(i): result_list[i] for i, gap in enumerate(gap_list)}
    result_list = [result_dict[str(i)] for i, gap in enumerate(gap_list)]


    ms.figure('Undulator wakefield measurements %s' % mat_file)
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    subplot = ms.subplot_factory(3,3)
    sp_ctr = 1

    sp_charge = subplot(sp_ctr, title='Current_profile')
    sp_ctr += 1
    sp_charge.plot(charge_xx*1e6, charge_profile)

    xlabel = 's [$\mu$m]'
    ylabel = 'w [kV/(pC$\cdot$m)]'
    ylabel2 = '$\Delta$ E [MeV]'
    ylabel3 = '$\Delta$ E [keV/m]'

    sp_wf_surface = subplot(sp_ctr, title='Surface wake', xlabel=xlabel, ylabel=ylabel)
    sp_ctr += 1

    sp_wf_res = subplot(sp_ctr, title='Resistive wake', xlabel=xlabel, ylabel=ylabel)
    sp_ctr += 1

    sp_wf_surface_W = subplot(sp_ctr, title='Surface wake convolved', xlabel=xlabel, ylabel=ylabel3)
    sp_ctr += 1

    sp_wf_res_W = subplot(sp_ctr, title='Resistive wake convolved', xlabel=xlabel, ylabel=ylabel3)
    sp_ctr += 1

    sp_chirp = subplot(sp_ctr, title='Combined effect', xlabel='s [$\mu$m]', ylabel=ylabel3)
    sp_ctr += 1

    sp_gap_effect = subplot(sp_ctr, title='Proj Energy loss', xlabel='Gap [mm]', ylabel='Energy loss [MeV]')
    sp_ctr += 1

    sp_espread = subplot(sp_ctr, title='Proj Espread increase [keV]', xlabel='Gap [mm]', ylabel='Espread change[MeV]')
    sp_ctr += 1

    eloss_surface = []
    eloss_surface2 = []
    eloss_ac = []

    espread_surface = []
    espread_ac = []

    W_combined = []

    for n_gap, (gap, wf_dict) in enumerate(zip(gap_list, result_list)):
        loss_surface = wf_dict['proj_Eloss_surface'] * L_factor
        loss_ac = wf_dict['proj_Eloss_ac'] * L_factor
        eloss_surface.append(loss_surface)
        eloss_ac.append(loss_ac)
        espread_surface.append(wf_dict['proj_Espread_surface'] * L_factor)
        espread_ac.append(wf_dict['proj_Espread_ac'] * L_factor)

        new_surf = uwf.surface_round_tube(charge_xx, gap/2, uwf.aramis_kappa*2, uwf.aramis_h, multi=True)[0]
        print('donezo')
        new_surf_W = uwf.convolve(new_surf, charge_profile)
        new_surf_eloss = np.sum(new_surf_W*charge_profile) / np.sum(charge_profile) * L_factor
        eloss_surface2.append(new_surf_eloss)

        label = '%.1f' % (gap*1e3)

        plot_xx = charge_xx * 1e6
        factor1 = 1e-3*1e-12
        factor2 = 1e-6
        factor3 = 1e-3

        comb = (wf_dict['W_ac']+wf_dict['W_surface'])
        comb2 = (wf_dict['W_ac']+new_surf_W)
        W_combined.append(comb)
        if n_gap % 3 == 0:
            sp_wf_surface.errorbar(plot_xx, wf_dict['w_surface']*factor1, label=label, yerr=wf_dict['w_surface_err']*factor1)
            color = sp_wf_res.plot(plot_xx, wf_dict['w_ac']*factor1, label=label)[0].get_color()
            sp_wf_surface.plot(plot_xx, new_surf*factor1, label=label, color=color, ls='--')
            sp_wf_res.plot(plot_xx, wf_dict['w_dc']*factor1, ls='--', color=color)
            sp_wf_surface_W.plot(plot_xx, wf_dict['W_surface']*factor3, label=label)
            sp_wf_surface_W.plot(plot_xx, new_surf_W*factor3, color=color, ls='--')
            sp_wf_res_W.plot(plot_xx, wf_dict['W_ac']*factor3, label=label)
            sp_wf_res_W.plot(plot_xx, wf_dict['W_dc']*factor3, ls='--', color=color)
            sp_chirp.plot(plot_xx, comb*factor3, label=label)
            sp_chirp.plot(plot_xx, comb2*factor3, color=color, ls='--')

    eloss_surface = np.array(eloss_surface)
    eloss_ac = np.array(eloss_ac)
    espread_surface = np.array(espread_surface)
    espread_ac = np.array(espread_ac)
    W_combined = np.array(W_combined)
    eloss_surface2 = np.array(eloss_surface2)

    eloss_surface -= eloss_surface.max()
    eloss_surface2 -= eloss_surface2.max()
    eloss_ac -= eloss_ac.max()

    sp_gap_effect.plot(gap_list*1e3, eloss_surface/1e6, label='Surface')
    sp_gap_effect.plot(gap_list*1e3, eloss_surface2/1e6, label='Surface 2')
    sp_gap_effect.plot(gap_list*1e3, eloss_ac/1e6, label='Resistive')
    sp_gap_effect.plot(gap_list*1e3, (eloss_ac+eloss_surface)/1e6, label='Combined')
    sp_gap_effect.plot(gap_list*1e3, (eloss_ac+eloss_surface2)/1e6, label='Combined 2')


    delta_E_screen = yy0 = dd['delta'].squeeze() * energy_eV
    yy = yy0.mean(axis=0)
    yy_err = yy0.std(axis=0)
    yy -= yy[0]
    sp_gap_effect.errorbar(gap_list*1e3, yy/1e6, yerr=yy_err/1e6, label='Screen')


    delta_E_bpm = yy0 = dd['delta1'].T.squeeze() * energy_eV
    yy = yy0.mean(axis=0)
    yy_err = yy0.std(axis=0)
    yy -= yy[0]
    sp_gap_effect.errorbar(gap_list*1e3, yy/1e6, yerr=yy_err/1e6, label='BPM')


    espread_fwhm = yy0 = dd['fwhm_delta'] * energy_eV
    yy = yy0.mean(axis=0)
    yy_err = yy0.std(axis=0)
    yy -= yy[0]
    espread_fwhm_plot = yy
    sp_espread.errorbar(gap_list*1e3, espread_fwhm_plot/1e6, yerr=yy_err/1e6, label='FWHM')


    espread_ac_plot = espread_ac - espread_ac[0]
    espread_surface_plot = espread_surface - espread_surface[0]
    sp_espread.plot(gap_list*1e3, espread_ac_plot/1e6, label='Resistive')
    sp_espread.plot(gap_list*1e3, espread_surface_plot/1e6, label='Surface')
    sp_espread.plot(gap_list*1e3, (espread_surface_plot+espread_ac_plot)/1e6, label='Combined')

    mean_x = np.mean(charge_xx)

    ms.figure('Debug fit')
    sp_ctr = 1
    sp_initial_chirp = subplot(sp_ctr, title='Initial chirp')
    sp_ctr += 1

    #sp_final_chirp = subplot(sp_ctr, title='Final chirp', sciy=True)
    #sp_ctr += 1

    sp_espread_fit = subplot(sp_ctr, title='Energy spread')
    sp_ctr += 1

    def fit_initial_chirp(gap_list, dE_ds, plot=False):
        init_chirp = dE_ds * (charge_xx - mean_x)
        #init_Espread = np.sqrt(np.sum(init_chirp**2*charge_profile)/np.sum(charge_profile) - (np.sum(init_chirp*charge_profile)/np.sum(charge_profile))**2)
        #init_Espread2 = uwf.calc_espread(init_chirp, charge_profile)

        if plot:
            label = '%.2e' % dE_ds
            sp_initial_chirp.plot(charge_xx, init_chirp/1e6, label=label)

        final_chirp_list = []
        for i, gap in enumerate(gap_list):
            final_chirp = init_chirp + W_combined[i]
            final_espread = uwf.calc_espread(final_chirp, charge_profile)
            final_chirp_list.append(final_espread)
            if plot:
                #sp_final_chirp.plot(charge_xx, final_chirp, label=label)
                pass
        #import pdb; pdb.set_trace()
        final_chirp_list = np.array(final_chirp_list)

        outp = final_chirp_list - final_chirp_list[0]
        if plot:
            sp_espread_fit.plot(gap_list*1e3, outp/1e6, label=label)
        return outp

    initial_guess = -1e8/charge_xx.max()
    fit_chirp = curve_fit(fit_initial_chirp, gap_list, espread_fwhm_plot, p0=[initial_guess])
    fit_chirp_yy0 = fit_initial_chirp(gap_list, fit_chirp[0])

    for deds in np.array([0, 0.8, 1, 1.2])*initial_guess:
        fit_chirp_yy = fit_initial_chirp(gap_list, deds, plot=True)
    sp_espread_fit.errorbar(gap_list*1e3, espread_fwhm_plot/1e6, yerr=yy_err/1e6, label='FWHM')

    sp_espread.plot(gap_list*1e3, fit_chirp_yy0/1e6, label='Fit')

    for sp_ in sp_wf_res, sp_wf_surface, sp_wf_res_W, sp_wf_surface_W, sp_espread, sp_gap_effect, sp_chirp:
        sp_.legend()


#ms.saveall('~/pcloud_share/presentations/022_uwf/011_show_all_meas', hspace=0.4, wspace=0.3)
plt.show()

