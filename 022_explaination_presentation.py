#import itertools
import copy; copy
import socket
import numpy as np; np
from scipy.constants import c
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

gaps = [10e-3, 10e-3]
beam_offsets = [4.75e-3, 0]
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.00
profile_cutoff = 0
len_profile = 1e3
struct_lengths = [1., 1.]
screen_bins = 100
smoothen = 0e-6
n_emittances = (300e-9, 300e-9)
n_particles = int(100e3)

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, n_particles=n_particles, screen_cutoff=screen_cutoff, screen_bins=screen_bins, smoothen=smoothen, profile_cutoff=profile_cutoff, len_screen=len_profile)
energy_eV = tracker.energy_eV


profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
profile_meas.center()

profile_gauss = tracking.get_gaussian_profile(sig_t, tt_halfrange, len_profile, charge, tracker.energy_eV, cutoff=profile_cutoff)

screen0 = tracker.matrix_forward(profile_meas, gaps, [0,0])['screen']
meas_screen = tracker.matrix_forward(profile_meas, gaps, beam_offsets)['screen']


ms.figure('')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1


sp_profile0 = subplot(sp_ctr, title='Beam profiles', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp_wake = subplot(sp_ctr, title='Wakefield', xlabel='t [fs]', ylabel='Single particle wake [V/C/m]')
sp_ctr += 1

sp_Wake = sp_wake.twinx()
sp_Wake.set_ylabel('Screen $\Delta$ x [mm]')
ms.sciy()

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Screen distribution (arb. units)')
sp_ctr += 1

sp_recon = subplot(sp_ctr, title='Reconstructed profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1


p = profile_meas
sp_recon.plot(p.time*1e15, p.current/p.integral, color='black', label='Real')

for p_ctr, (p, label) in enumerate([(profile_meas, 'Measured'), (profile_gauss, 'Gaussian')]):
    p.center()

    sp_profile0.plot(p.time*1e15, p.current/p.integral, label='%i' % (p.gaussfit.sigma*1e15))

    wake = p.calc_wake(gaps[0], beam_offsets[0], struct_lengths[0])
    wake_effect = p.wake_effect_on_screen(wake, tracker.calcR12()[n_streaker])

    wake_yy = wake['dipole']['single_particle_wake']
    wake_xx = wake['input']['charge_xx']/c

    Wake_yy = wake['dipole']['wake_potential']

    if p_ctr == 0:
        sp_wake.plot((wake_xx+wake_effect['t'].min())*1e15, wake_yy, color='black')
    sp_Wake.plot(wake_effect['t']*1e15, wake_effect['x']*1e3, label=label)

    screen = tracker.matrix_forward(p, gaps, beam_offsets)['screen']

    sp_screen.plot(screen.x*1e3, screen.intensity/screen.integral, label=label)


    bp_recon = tracker.track_backward(meas_screen, screen0, wake_effect)


    sp_recon.plot(bp_recon.time*1e15, bp_recon.current/bp_recon.integral, label=label + '/ %i' % (bp_recon.gaussfit.sigma*1e15))

    if True:
        screen_forward = tracker.matrix_forward(bp_recon, gaps, beam_offsets)['screen']

        sp_screen.plot(screen_forward.x*1e3, screen_forward.intensity/screen_forward.integral, label=label+' rec', ls='--')


sp_screen.set_ylim(0, 2e3)


sp_profile0.legend(title='Duration [fs]')
sp_screen.legend()
sp_Wake.legend()
sp_recon.legend(title='Duration [fs]')


ms.saveall('./group_metting_2020-11-17/explain')


plt.show()
