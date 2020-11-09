import socket
import numpy as np; np
import matplotlib.pyplot as plt

import elegant_matrix
import tracking
#from scipy.optimize import curve_fit

import myplotstyle as ms

plt.close('all')


hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

#energy_eV = 6.14e9
gaps = [10e-3, 10e-3]
beam_offsets = [4.7e-3, 0.]
fit_order = 4
sig_t = 30e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
backtrack_cutoff = 0.05
len_profile = 1e3
struct_lengths = [1., 1.]

if hostname == 'desktop':
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/storage/data_2020-02-03/Bunch_length_meas_2020-02-03_15-59-13.h5'
else:
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'
    bl_meas_file = '/sf/data/measurements/2020/02/03/Bunch_length_meas_2020-02-03_15-59-13.h5'


tracker = tracking.Tracker(magnet_file, timestamp, charge, struct_lengths, energy_eV='file')
energy_eV = tracker.energy_eV

profile_meas = tracking.profile_from_blmeas(bl_meas_file, tt_halfrange, energy_eV, subtract_min=False)

profile_back = tracker.forward_and_back(profile_meas, profile_meas, gaps, beam_offsets, 0)


#track_dict_forward = tracker.elegant_forward(profile_meas, gaps, beam_offsets, [1, 1])
#track_dict_forward0 = tracker.elegant_forward(profile_meas, gaps, [0,0], [1, 1])
#
#wf_dict = profile_meas.calc_wake(gaps[0], beam_offsets[0], 1.)
#wake_effect = profile_meas.wake_effect_on_screen(wf_dict, track_dict_forward0['r12_dict'][0])
#profile_back = tracker.track_backward(track_dict_forward, track_dict_forward0, wake_effect)
#profile_back.reshape(len_profile)

ms.figure('Back and forward with real profile')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp = subplot(sp_ctr, title='Profiles')

for bp, label in [(profile_meas, 'Measured'), (profile_back, 'Reconstructed')]:
    norm = np.trapz(bp.current, bp.time)
    sp.plot(bp.time*1e15, bp.current/norm, label=label)

sp.legend()

plt.show()



