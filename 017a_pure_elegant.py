import socket
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

import wf_model
import elegant_matrix
import data_loader
from gaussfit import GaussFit

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

energy_eV = 6.14e9
gap = 10e-3
beam_offset = 4.7e-3
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15

investigate_wake = False
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)

hostname = socket.gethostname()
if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'

### Generate Gaussian beam to calculate initial wake potential

tt_dict, wake_dict, current_dict = {}, {}, {}
for sig_t2 in (30e-15, sig_t, 50e-15):
    time_gauss = np.linspace(-tt_halfrange, tt_halfrange, 1000)

    current_gauss = np.exp(-(time_gauss-np.mean(time_gauss))**2/(2*sig_t2**2))
    current_gauss[current_gauss<0.001*current_gauss.max()] = 0
    current_gauss = current_gauss * 200e-12/np.sum(current_gauss)
    current_dict[sig_t2] = current_gauss

    wf_calc = wf_model.WakeFieldCalculator((time_gauss-time_gauss.min())*c, current_gauss, energy_eV, 1.)
    wf_dict = wf_calc.calc_all(gap/2., 1., beam_offset=beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=False)

    gaussian_wake_tt = time_gauss
    gaussian_wake = wf_dict['dipole']['wake_potential']

    tt_dict[sig_t2] = time_gauss
    wake_dict[sig_t2] = gaussian_wake

time_gauss = tt_dict[sig_t]
gaussian_wake = wake_dict[sig_t]
current_gauss0 = current_dict[sig_t]


if investigate_wake:

    # Forward track with elegant and Gaussian beam
    simulator = elegant_matrix.get_simulator(magnet_file)
    #mat_dict, disp_dict = simulator.get_elegant_matrix(0, timestamp)

    sim, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time_gauss, current_gauss, timestamp, (gap, 10e-3), (beam_offset, 0), energy_eV, linearize_twf=False)
    mat_dict, _ = simulator.get_elegant_matrix(0, timestamp)
    r12 = mat_dict['SARBD02.DSCR050'][0,1]

    single_wake2 = np.interp(time_gauss, wf_dicts[0]['t'], wf_dicts[0]['WX'])
    convoluted_wake2 = np.convolve(current_gauss, single_wake2)[:len(current_gauss)]


    ### Look at used wakefield
    ms.figure('Wakefields')
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_wake = subplot(sp_ctr, title='Initial wake from Gaussian $\sigma_t$=%i fs' % (sig_t*1e15), xlabel='t [fs]', ylabel='kV/m')
    sp_ctr += 1

    sp_wake.plot(gaussian_wake_tt*1e15, gaussian_wake/1e3, label='Wake', lw=3)
    sp_wake.plot(time_gauss*1e15, -convoluted_wake2/1e3, label='Wake 2')

    for fit_order2 in [fit_order]:
        poly_fit = np.poly1d(np.polyfit(gaussian_wake_tt, gaussian_wake, fit_order2))
        fitted_wake = poly_fit(gaussian_wake_tt)
        sp_wake.plot(gaussian_wake_tt*1e15, fitted_wake/1e3, label='Fit %ith order' % fit_order2)

    sp_wake.legend()


### Forward track with elegant for measured beam profile

ms.figure('Forward and backward tracking - Ignore natural beamsize (200 nm emittance)')
subplot = ms.subplot_factory(2,3)
sp_ctr = 1


bl_meas = data_loader.load_blmeas(bl_meas_file)
simulator = elegant_matrix.get_simulator(magnet_file)

time_meas0 = bl_meas['time_profile1']
current_meas0 = bl_meas['current1']
current_meas0 *= 200e-12/current_meas0.sum()
gf_blmeas = GaussFit(time_meas0, current_meas0)
time_meas0 -= gf_blmeas.mean

time_meas1 = np.arange(-tt_halfrange, time_meas0.min(), np.diff(time_meas0).mean())
time_meas2 = np.arange(time_meas0.max(), tt_halfrange, np.diff(time_meas0).mean())
time_meas = np.concatenate([time_meas1, time_meas0, time_meas2])
current_meas = np.concatenate([np.zeros_like(time_meas1), current_meas0, np.zeros_like(time_meas2)])

sim1, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time_meas, current_meas, timestamp, (gap, 10e-3), (beam_offset, 0), energy_eV, linearize_twf=False)

sim0, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time_meas, current_meas, timestamp, (gap, 10e-3), (0, 0), energy_eV, linearize_twf=False)
r12 = mat_dict['SARBD02.DSCR050'][0,1]
mean_X0 = sim0.watch[-1]['x'].mean()

sp = subplot(sp_ctr, title='Current profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp.plot(time_meas*1e15, current_meas/current_meas.max(), label='Measured')
sp.plot(time_gauss*1e15, current_gauss/current_gauss.max(), label='Initial guess')

sp.legend()

sp = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Current (arb. units)')
sp_ctr += 1

for sim, label in [(sim0, 'No streaking'), (sim1, 'Streaking')]:
    w = sim.watch[-1]

    hist, bin_edges = np.histogram(w['x'], bins=200, density=True)
    screen_xx = -(bin_edges[:-1] - mean_X0)
    screen_hist = hist / hist.max()

    sp.step(screen_xx*1e3, screen_hist, label=label)
sp.legend()


### Backward tracking

xx_wake = gaussian_wake / energy_eV * r12

real_wf_calc = wf_model.WakeFieldCalculator((time_meas-time_meas.min())*c, current_meas, energy_eV, 1)
real_wake = -real_wf_calc.calc_all(gap/2., 1, beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=False)['dipole']['wake_potential']

#real_wake_spw2 = np.interp(time_meas, real_wake_tt, real_wake_spw)
#real_wake = np.convolve(current_meas, real_wake_spw2)[:len(current_meas)]
xx_real_wake = real_wake / energy_eV * r12

sp = subplot(sp_ctr, title='Guessed wake effect', xlabel='t [fs]', ylabel='x [mm]')
sp_ctr += 1

sp.plot(time_meas*1e15, -xx_real_wake*1e3, label='Real')

for sig_t2 in (30e-15, sig_t, 50e-15):
    xx_wake2 = wake_dict[sig_t2] / energy_eV * r12
    sp.plot(tt_dict[sig_t2]*1e15, xx_wake2*1e3, label='Gaussian %i fs' % (sig_t2*1e15))

sp.legend()

t_interp = np.interp(screen_xx, xx_wake, gaussian_wake_tt)
charge_interp = -screen_hist[1:] / np.diff(t_interp)
charge_interp[charge_interp == np.inf] = np.nan
charge_interp[charge_interp == -np.inf] = np.nan

sp = subplot(sp_ctr, title='Backpropagated time', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

charge_corrected = charge_interp/np.nanmax(charge_interp)
charge_uncorrected = screen_hist/screen_hist.max()

sp.step(t_interp[1:]*1e15, charge_corrected, label='Corrected')
sp.step(t_interp*1e15, charge_uncorrected, label='Uncorrected')
sp.legend(title='Nonlinearity')

sp = subplot(sp_ctr, title='Comparison', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

for time, current, label in [(t_interp[1:], charge_corrected, 'Backtracked'), (time_meas, current_meas, 'Input'), (time_gauss, current_gauss0, 'Initial guess'),]:

    nan_arr = np.logical_or(np.isnan(time), np.isnan(current))
    gf = GaussFit(time[~nan_arr], current[~nan_arr])
    time2 = time - gf.mean

    color = sp.plot(time2*1e15, current/np.nanmax(current), label=label+' %i' % (gf.sigma*1e15))[0].get_color()
    sp.plot(gf.xx*1e15, gf.reconstruction/gf.reconstruction.max(), color=color, ls='--')

sp.legend(title='Gaussian RMS [fs]')

#ms.saveall('/tmp/017a_pure_elegant', transparent=False)

plt.show()


