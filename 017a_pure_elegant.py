import socket
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

import wf_model
import elegant_matrix
import data_loader

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

energy_eV = 6.14e9
gap = 10e-3
beam_offset = 4.7e-3
fit_order = 4
zoom_sig = 3

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

# approximate 40 fs beam
sig_t = 40e-15
time = np.linspace(0., 400e-15, 1000)

curr = np.exp(-(time-np.mean(time))**2/(2*sig_t**2))
curr[curr<0.001*curr.max()] = 0
curr = curr * 200e-12/np.sum(curr)

wf_calc = wf_model.WakeFieldCalculator(time*c, curr, energy_eV, 1.)
wf_dict = wf_calc.calc_all(gap/2., 1., beam_offset=beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=False)

wake_zz = wf_dict['input']['charge_xx']
convoluted_wake = wf_dict['dipole']['wake_potential']


if investigate_wake:

    # Forward track with elegant and Gaussian beam
    simulator = elegant_matrix.get_simulator(magnet_file)
    #mat_dict, disp_dict = simulator.get_elegant_matrix(0, timestamp)

    sim, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time, curr, timestamp, (gap, 10e-3), (beam_offset, 0), energy_eV, linearize_twf=False)
    r12 = mat_dict['SARBD02.DSCR050'][0,1]

    single_wake2 = np.interp(time, wf_dicts[0]['t'], wf_dicts[0]['WX'])
    convoluted_wake2 = np.convolve(curr, single_wake2)[:len(curr)]


    ### Look at used wakefield
    ms.figure('Wakefields')
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_wake = subplot(sp_ctr, title='Initial wake from Gaussian $\sigma_t$=%i fs' % (sig_t*1e15), xlabel='s [$\mu$m]', ylabel='kV/m')
    sp_ctr += 1

    sp_wake.plot(wake_zz*1e6, convoluted_wake/1e3, label='Wake', lw=3)
    sp_wake.plot(time*c*1e6, -convoluted_wake2/1e3, label='Wake 2')

    for fit_order2 in [fit_order]:
        poly_fit = np.poly1d(np.polyfit(wake_zz, convoluted_wake, fit_order2))
        fitted_wake = poly_fit(wake_zz)
        sp_wake.plot(wake_zz*1e6, fitted_wake/1e3, label='Fit %ith order' % fit_order2)

    sp_wake.legend()


### Forward track with elegant for measured beam profile

ms.figure('Forward and backward tracking')
subplot = ms.subplot_factory(2,3)
sp_ctr = 1


bl_meas = data_loader.load_blmeas(bl_meas_file)
simulator = elegant_matrix.get_simulator(magnet_file)

time_meas = bl_meas['time_profile1']
current_meas = bl_meas['current1']
current_meas *= 200e-12/current_meas.sum()

sim1, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time_meas, current_meas, timestamp, (gap, 10e-3), (beam_offset, 0), energy_eV, linearize_twf=False)

sim0, mat_dict, wf_dicts, disp_dict = simulator.simulate_streaker(time_meas, current_meas, timestamp, (gap, 10e-3), (0, 0), energy_eV, linearize_twf=False)
r12 = mat_dict['SARBD02.DSCR050'][0,1]
mean_X0 = sim0.watch[-1]['x'].mean()

sp = subplot(sp_ctr, title='Current profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp.plot(time_meas*1e15, current_meas/current_meas.max())

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

xx_wake = convoluted_wake / energy_eV * r12

sp = subplot(sp_ctr, title='Guessed wake effect', xlabel='t [fs]', ylabel='x [mm]')
sp_ctr += 1
wake_tt = wake_zz/c

sp.plot(wake_tt*1e15, xx_wake*1e3)

t_interp = np.interp(screen_xx, xx_wake, wake_tt)
charge_interp = -screen_hist[1:] / np.diff(t_interp)
charge_interp[charge_interp == np.inf] = np.nan

sp = subplot(sp_ctr, title='Backpropagated time', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

charge_corrected = charge_interp/np.nanmax(charge_interp)
charge_uncorrected = screen_hist/screen_hist.max()

sp.step(t_interp[1:]*1e15, charge_corrected, label='Corrected')
sp.step(t_interp*1e15, charge_uncorrected, label='Uncorrected')
sp.legend()

sp = subplot(sp_ctr, title='Comparison', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp.plot(time_meas*1e15, current_meas/current_meas.max(), label='Input')
sp.plot(t_interp[1:]*1e15, charge_corrected, label='Backtracked')
sp.legend()

plt.show()


