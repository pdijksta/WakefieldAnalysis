import pickle
import numpy as np
from scipy.constants import c
from socket import gethostname
from scipy.io import savemat

import wf_model
import tracking

struct_lengths = [1., 1.]
timestamp = 1601761132
n_particles = int(1e5)
n_emittances = [500e-9, 500e-9]
screen_bins = 500
screen_cutoff = 1e-2
smoothen = 30e-6
profile_cutoff = 0
timestamp = 1601761132
gap_correcting_summand = 0
gaps = [10e-3, 10e-3 + gap_correcting_summand]
#offset_correcting_summand = 10e-6
offset_correcting_summand = 0
mean_offset = 0.472*1e-3 + offset_correcting_summand
n_streaker = 1
tt_halfrange = 200e-15
bp_smoothen = 1e-15
quad_wake = False
len_profile = int(2e3)



hostname = gethostname()
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

magnet_file = archiver_dir + 'archiver_api_data/2020-10-03.h5'

beam_offset = 4.692e-3
semigap = 5e-3

s_arr = np.linspace(0, 400e-15*c, int(1e5))

spw_dipole = wf_model.wxd(s_arr, semigap, beam_offset)
spw_quadrupole = wf_model.wxq(s_arr, semigap, beam_offset)

tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, bp_smoothen=bp_smoothen, quad_wake=quad_wake)

streaker_matrices = tracker.simulator.get_streaker_matrices(timestamp)

streaker_matrix = streaker_matrices['s2_to_screen']

with open('./avg_screen.pkl', 'rb') as f:
    avg_screen = pickle.load(f)


with open('./tdc_profile.pkl', 'rb') as f:
    tdc_profile = pickle.load(f)

output = {
        'semigap': semigap,
        'beam_offset': beam_offset,
        's_arr': s_arr,
        'spw_dipole': spw_dipole,
        'spw_quadrupole': spw_quadrupole,
        'screen_x': avg_screen.x,
        'screen_intensity': avg_screen.intensity,
        'tdc_time': tdc_profile.time,
        'tdc_current': tdc_profile.current,
        'streaker_to_screen': streaker_matrix,
        }

savemat('./streaker_info.mat', output)

