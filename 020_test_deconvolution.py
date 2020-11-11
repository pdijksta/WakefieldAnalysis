#import itertools
#import copy
import pickle
import socket
import numpy as np; np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve

import elegant_matrix
import tracking
#from scipy.optimize import minimize; minimize

import myplotstyle as ms

plt.close('all')


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

#energy_eV = 6.14e9
gaps = [10e-3, 10e-3]
beam_offsets = [4.7e-3, 0]
n_streaker = 0
fit_order = 4
sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
screen_cutoff = 0.03
profile_cutoff = 0.00
len_profile = 1e3
struct_lengths = [1., 1.]
n_bins=500
smoothen = 0e-6
n_emittances = (800e-9, 800e-9)
n_particles = int(100e3)

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


if False:
    tracker0 = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=(1e-9, 1e-9), n_bins=n_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff)
    tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, energy_eV='file', n_emittances=n_emittances, n_bins=n_bins, n_particles=n_particles, smoothen=smoothen, profile_cutoff=profile_cutoff, screen_cutoff=screen_cutoff)
    energy_eV = tracker.energy_eV

    profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, charge, energy_eV, subtract_min=False)
    profile_meas.cutoff(profile_cutoff)
    profile_meas.reshape(len_profile)

    fab_dict = tracker.forward_and_back(profile_meas, profile_meas, gaps, beam_offsets, n_streaker)
    fab_dict0 = tracker0.forward_and_back(profile_meas, profile_meas, gaps, beam_offsets, n_streaker)
    screen_meas = fab_dict['track_dict_forward']['screen']
    screen_meas0 = fab_dict['track_dict_forward0']['screen']


    screen_meas00 = fab_dict0['track_dict_forward']['screen']
    screen_meas000 = fab_dict0['track_dict_forward0']['screen']


    with open('./screens.pkl', 'wb') as f:
        pickle.dump({
            'screen_meas0': screen_meas0,
            'screen_meas00': screen_meas00,
            'screen_meas000': screen_meas000,
            'screen_meas': screen_meas,
            }, f)
else:
    with open('./screens.pkl', 'rb') as f:
        s_dict = pickle.load(f)

    screen_meas = s_dict['screen_meas']
    screen_meas0 = s_dict['screen_meas0']
    screen_meas00 = s_dict['screen_meas00']
    screen_meas000 = s_dict['screen_meas000']

beamsize = screen_meas0.gaussfit.sigma

ms.figure('Screens')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1


sp0 = subplot(1, title='Screens')

for s, label in [(screen_meas0, 'No streaking large emittance'), (screen_meas, 'Streaking large emittance'), (screen_meas000, 'No streaking small emittance'), (screen_meas00, 'Streaking small emittance')]:
    sp0.plot(s.x, s.intensity/s.intensity.max(), label=label)

sp = subplot(2, title='Screens streaked')

for s, label in [(screen_meas, 'Large emittance'), (screen_meas00, 'Small emittance')]:
    sp.plot(s.x, s.intensity/s.integral, label=label)

xx_arr = screen_meas.x
gauss = np.exp(-(xx_arr-np.mean(xx_arr))**2/(2*beamsize**2))/(np.sqrt(2*np.pi)*beamsize)
cut_gauss = gauss[gauss>1]
cut_xx = xx_arr[gauss>1]

cut_gauss /= np.sum(cut_gauss)

sp_gauss = subplot(3, title='Gauss')
sp_gauss.plot(xx_arr, gauss)
sp_gauss.plot(cut_xx, cut_gauss)

deconvolved, remainder = deconvolve(screen_meas.x, cut_gauss)

xxd = xx_arr[:len(deconvolved)]
deconvolved_norm = deconvolved/np.trapz(deconvolved, xx_arr[:len(deconvolved)])

sp.plot(xxd, deconvolved_norm, label='Deconvolved')

sp0.plot(cut_xx, cut_gauss/cut_gauss.max(), label='Gauss')
sp0.legend()



sp.legend()

plt.show()






