#import itertools
import copy; copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

gaps = [10e-3, 10e-3]
beam_offset_list = ([4.7e-3, 0], [4.75e-3, 0], [4.65e-3, 0])
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.02
profile_cutoff = 0
len_profile = 1e3
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = (300e-9, 300e-9)
n_particles = int(100e3)

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, screen_bins=screen_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff, len_screen=len_profile)
energy_eV = tracker.energy_eV


profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=profile_cutoff)
profile_meas.center()

fdict0 = tracker.elegant_forward(profile_meas, gaps, [0., 0.])

fdict1 = tracker.matrix_forward(profile_meas, gaps, [0., 0.])

screen0 = fdict0['screen']
screen1 = fdict1['screen']


ms.figure('Investigate screen')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Screen without streaking')
sp_ctr += 1

screen0.plot_standard(sp, label='Elegant')

sp.legend()

sp = subplot(sp_ctr, title='Screen without streaking')
sp_ctr += 1

screen1.plot_standard(sp, label='Matrix')

sp.legend()


plt.show()







