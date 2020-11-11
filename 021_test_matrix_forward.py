import time
import socket
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import wf_model

import elegant_matrix
import tracking

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)

gaps = [10e-3, 10e-3]
beam_offsets = [4.7e-3, 0]
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.00
profile_cutoff = 0.00
len_profile = 1e3
struct_lengths = [1., 1.]
n_bins=500
smoothen = 0e-6
n_emittances = (300e-9, 300e-9)
n_particles = int(100e3)


if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'

tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles=n_particles, n_emittances=n_emittances, n_bins=n_bins, screen_cutoff=screen_cutoff, smoothen=smoothen, profile_cutoff=profile_cutoff, len_screen=len_profile)

profile =tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, tracker.energy_eV, subtract_min=True)

tmat0 = time.time()
simulator = elegant_matrix.get_simulator(magnet_file)


mat_dict = simulator.get_streaker_matrices(timestamp)

s1 = mat_dict['start_to_s1']

beta_x = 5.067067
beta_y = 16.72606
alpha_x = -0.5774133
alpha_y = 1.781136

watch, sim = elegant_matrix.gen_beam(300e-9, 300e-9, alpha_x, beta_x, alpha_y, beta_y, 6e9/511e3, 40e-15, n_particles)

curr = profile.current
tt = profile.time
integrated_curr = np.cumsum(curr)
integrated_curr /= integrated_curr.max()

randoms = np.random.rand(n_particles)
interp_tt = np.interp(randoms, integrated_curr, tt)
interp_tt -= interp_tt.min()


p_arr = np.ones_like(watch['x'])*tracker.energy_eV/511e3
#t_arr =
beam_start = np.array([watch['x'], watch['xp'], watch['y'], watch['yp'], interp_tt, p_arr])
beam_before_s1 = np.matmul(s1, beam_start)


#wf_dict_s1 = profile.calc_wake(gaps[0], beam_offsets[0], struct_lengths[0])

#wf_xx = np.linspace(interp_tt.min(), interp_tt.max(), 1000)*1.1*c
#wf_xx -= wf_xx.min()
wf_hist, bin_edges = np.histogram(interp_tt, bins=1000)
wf_hist = wf_hist / np.sum(wf_hist) * charge
wake_tt = (bin_edges[1:] + bin_edges[:-1])/2
wake_tt -= wake_tt.min()
wf_xx = wake_tt*c
wf_calc = wf_model.WakeFieldCalculator(wf_xx, wf_hist, tracker.energy_eV, Ls=struct_lengths[0])
wf_dict = wf_calc.calc_all(gaps[0]/2., 1., beam_offsets[0], calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=False)
wake = wf_dict['dipole']['wake_potential']




#wake = wf_dict_s1['dipole']['wake_potential']
#wake_tt = wf_dict_s1['input']['charge_xx']/c

wake_energy = np.interp(beam_before_s1[4,:], wake_tt, wake)
delta_xp = wake_energy/tracker.energy_eV


beam_after_s1 = np.copy(beam_before_s1)
beam_after_s1[1,:] += delta_xp

beam_before_s2 = np.matmul(mat_dict['s1_to_s2'], beam_after_s1)

beam_after_s2 = beam_before_s2

beam_at_screen = np.matmul(mat_dict['s2_to_screen'], beam_after_s2)
beam0_at_screen = mat_dict['s2_to_screen'] @ mat_dict['s1_to_s2'] @ beam_before_s1
beam_at_screen[0] -= beam0_at_screen[0].mean()

screen_matrix = tracking.getScreenDistributionFromPoints(beam_at_screen[0,:], n_bins=n_bins)


tmat1 = tele0 = time.time()

forward_dict = tracker.elegant_forward(profile, gaps, beam_offsets)
sim = forward_dict['sim']
sdds_wake = forward_dict['sdds_wakes'][0]
screen_elegant = forward_dict['screen']

tt_sdds = sdds_wake['t']
current_interp = np.interp(tt_sdds, profile.time-profile.time.min(), profile.current, right=0)
convoluted_wake_ele = np.convolve(sdds_wake['WX'], current_interp)[:len(sdds_wake['WX'])]

tele1 = time.time()

print('Time elegant: %.3f' % (tele1-tele0))
print('Time matrix: %.3f' % (tmat1-tmat0))

screen_module = tracker.matrix_forward(profile, gaps, beam_offsets)['screen']


ms.figure('Compare matrix with elegant tracking')

subplot = ms.subplot_factory(2,2)

sp = subplot(1, title='Screen distribution')

for s, label in [(screen_matrix, 'Matrix'), (screen_elegant, 'Elegant'), (screen_module, 'Module')]:
    s.normalize()
    sp.plot(s.x*1e3, s.intensity, label=label)

sp.legend()

sp = subplot(2, title='Wakes', xlabel='t [s]', ylabel='E [MeV]')

sp.plot(tt_sdds, convoluted_wake_ele/1e6, label='Elegant')
sp.plot(wake_tt, wake/1e6, label='Matrix')

sp.legend()


subplot = ms.subplot_factory(2,3)
for beam, label in [
        (beam_start, 'begin'),
        (beam_before_s1, 'before s1'),
        (beam_after_s1, 'after s1'),
        (beam_before_s2, 'before s2'),
        (beam_at_screen, 'at screen'),
        ]:
    ms.figure(label)
    sp_ctr = 1
    for i, label in zip(range(6), ['x', 'xp', 'y', 'yp', 't', 'p']):
        sp = subplot(sp_ctr, title=label, scix=True, sciy=True)
        sp_ctr += 1
        sp.hist(beam[i], bins=n_bins)




plt.show()

