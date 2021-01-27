import re
import numpy as np
import socket
import pickle
import matplotlib.pyplot as plt

import myplotstyle as ms
import tracking
import elegant_matrix


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

# 3rd October
if hostname == 'desktop':
    dirname1 = '/storage/data_2020-10-03/'
    dirname2 = '/storage/data_2020-10-04/'
    archiver_dir = '/storage/Philipp_data_folder/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2020/10/03/'
    dirname2 = '/sf/data/measurements/2020/10/04/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2020-10-03/'
    dirname2 = '/home/work/data_2020-10-04/'
    archiver_dir = '/home/work/'

re_file = re.compile('Passive_data_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).mat')
def get_timestamp(filename):
    match = re_file.match(filename)
    args = [int(x) for x in match.groups()]
    if match is None:
        print(filename)
        raise ValueError
    return elegant_matrix.get_timestamp(*args)




tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 0.03
profile_cutoff = 0.00
len_profile = 5e3
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = (1500e-9, 500e-9)
n_particles = int(100e3)
n_streaker = 1
flip_measured = False

mean_struct2 = 472e-6 # see 026_script
gap2_correcting_summand = -40e-6
sig_t_range = np.arange(20, 50.01, 1)*1e-15
gaps = [10e-3, 10e-3+gap2_correcting_summand]


self_consistent = True

plt.close('all')

ms.figure('Compare')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen')
sp_ctr += 1


sp_profile = subplot(sp_ctr, title='Profile')
sp_ctr += 1



with open('./025.pkl', 'rb') as f:
    d1 = pickle.load(f)

with open('./025d.pkl', 'rb') as f:
    d2 = pickle.load(f)


for d, label in [(d1, '025'), (d2, '025d')]:
    meas_screen = d['meas_screen']
    best_screen = d['best_screen']
    best_profile = d['best_profile']

    meas_screen.crop()

    meas_screen.plot_standard(sp_screen, label=label+' Meas')
    #best_screen.plot_standard(sp_screen, label=label+' Rec')
    best_profile.plot_standard(sp_profile, label=label, center_max=True, norm=True)
    file_ = 'Passive_data_20201003T231958.mat'
    timestamp = get_timestamp(file_)
    tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, compensate_negative_screen=True)
    beam_offsets = [0.0, 0.004692]
    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=self_consistent)

    best_profile2 = gauss_dict['reconstructed_profile']
    best_profile2.plot_standard(sp_profile, label=label +' 2', center_max=True, norm=True)

    if not self_consistent:

        best_profile3 = gauss_dict['final_profile']
        best_profile3.plot_standard(sp_profile, label=label +' final', center_max=True, norm=True)



sp_screen.legend()
sp_profile.legend()




plt.show()




