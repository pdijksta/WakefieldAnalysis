import sys
import os
new_path = os.path.dirname(__file__)+'/..'
if new_path not in sys.path:
    sys.path.append(new_path)
import argparse
import pickle
import socket
import matplotlib.pyplot as plt
import numpy as np
#from scipy.constants import c

import WakefieldAnalysis.elegant_matrix as elegant_matrix
#import WakefieldAnalysis.image_and_profile as iap
import WakefieldAnalysis.myplotstyle as ms
import WakefieldAnalysis.image_and_profile as iap
#import WakefieldAnalysis.gaussfit as gaussfit

parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--savedir', type=str)
parser.add_argument('--savepkl', action='store_true')
args = parser.parse_args()

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp/elegant')

hostname = socket.gethostname()

if hostname == 'desktop':
    archiver_dir = '/storage/Philipp_data_folder/archiver_api_data/'
    data_dir = '/storage/data_2021-03-16/'
elif hostname == 'pc11292.psi.ch':
    pass
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-03-16/'
    archiver_dir = '/mnt/data/archiver_api_data/'

filename = os.path.join(os.path.dirname(__file__), './plots_for_spie2.pkl')

with open(filename, 'rb') as f:
    pkl_dict = pickle.load(f)

tracker = pkl_dict['tracker']
profile_meas = pkl_dict['profile_meas']
profile_meas.reshape(int(5e3))
tracker.n_particles = int(1e6)

elegant_matrix.set_tmp_dir('~/tmp/elegant')
tracker.set_bs_at_streaker()
print(tracker)
bs_at_streaker = tracker.bs_at_streaker[1]

#tracker.override_quad_beamsize = True
#tracker.quad_x_beamsize = [0, bs_at_streaker/2]
tracker.n_emittances = [200e-9, 200e-9]
#tracker.split_streaker = 5
#tracker.wake2d = True

ms.figure('For paper', figsize=(6, 5))
subplot = ms.subplot_factory(2, 2, grid=False)
sp_ctr = 1

sp_resolution0 = subplot(sp_ctr, xlabel='t [fs]', ylabel='R [fs]', title='Resolution')
sp_ctr += 1



ms.figure('Current profile')
subplot = ms.subplot_factory(3,3)
sp_ctr = 1

sp_current = subplot(sp_ctr, title='Current profile', xlabel='t [fs]', ylabel='Current [kA]')
sp_ctr += 1

sp_wake = sp_current.twinx()
sp_wake.set_ylabel('Wake [kV/m]')

profile_meas.plot_standard(sp_current, color='black', label='Current')

semigap = 5e-3
gaps = [0, 2*semigap]
beam_offset = 4.7e-3
beam_offsets = [0, beam_offset]
struct_length = 1

wf_calc = profile_meas.calc_wake(semigap*2, beam_offset, struct_length)
_, wf_x = profile_meas.get_x_t(gaps[1], beam_offset, struct_length, tracker.calcR12()[1])
wf_t = profile_meas.time
dxdt = np.diff(wf_x)/np.diff(wf_t)
dx_dt_t = wf_t[:-1]

sp_dxdt = subplot(sp_ctr, title='Streaking strength', xlabel='t [fs]', ylabel='dx/dt [$\mu$m/fs]')
sp_ctr += 1
sp_dxdt.plot(dx_dt_t*1e15, dxdt*(1e6/1e15))

dipole_wake = wf_calc['dipole']['wake_potential']
quadrupole_wake = wf_calc['quadrupole']['wake_potential']

sp_wake.plot(profile_meas.time*1e15, dipole_wake*1e-3, label='$w_d$')
sp_wake.plot(profile_meas.time*1e15, quadrupole_wake*bs_at_streaker*1e-3, label='$w_d+w_q$')

ms.comb_legend(sp_current, sp_wake)


sp_beamsize = subplot(sp_ctr, title='Beamsize', xlabel='t [fs]', ylabel='$\sigma_x$ [$\mu$m]')
sp_ctr += 1

#sp_ss = subplot(sp_ctr, title='Streaking strength')
#sp_ctr += 1

sp_resolution = subplot(sp_ctr, title='Resolution', xlabel='t [fs]', ylabel='R [fs]')
sp_ctr += 1



for quad_wake in (False, True):
    tracker.quad_wake = quad_wake
    forward_dict = tracker.matrix_forward(profile_meas, gaps, beam_offsets)
    beam = forward_dict['beam_at_screen']
    if quad_wake:
        label = '$w_d+w_q$'
    else:
        label = '$w_d$'

    sp = subplot(sp_ctr, title=label, xlabel='t [fs]', ylabel='x [$\mu$m]', grid=False)
    sp_ctr += 1
    beam_t = beam[-2]
    hist, xedges, yedges = np.histogram2d(beam_t-beam_t.mean(), beam[0], bins=(250,200))
    t_axis = (xedges[1:] + xedges[:-1])/2.
    x_axis = (yedges[1:] + yedges[:-1])/2.
    x_axis2 = np.ones_like(hist)*x_axis
    extent=[xedges[0]*1e15, xedges[-1]*1e15, yedges[0]*1e6, yedges[-1]*1e6]
    sp.imshow(hist, extent=extent, aspect='auto')

    current_t = hist.sum(axis=1)
    #current_x = hist.sum(axis=0)
    mean_x = np.sum(hist*x_axis, axis=1) / current_t
    mean_x2 = np.ones_like(hist)*mean_x[:,np.newaxis]
    beamsize_sq = np.sum(hist*(x_axis2 - mean_x2)**2, axis=1) / current_t
    beamsize = np.sqrt(beamsize_sq)

    #beamsize2 = np.zeros_like(beamsize)
    #for n_t, t in enumerate(t_axis):
    #    beamsize2[n_t] = gaussfit.GaussFit(x_axis, hist[n_t,:], fit_const=False).sigma
    #sp_beamsize.plot(t_axis, beamsize2*1e6)

    sp_beamsize.plot(t_axis*1e15, beamsize*1e6, label=label)

    streaking_strength = np.interp(t_axis, dx_dt_t, dxdt)
    #sp_ss.plot(t_axis*1e15, streaking_strength)

    resolution = beamsize / streaking_strength
    resolution2 = iap.calc_resolution(profile_meas, gaps[1], beam_offsets[1], 1, tracker, 1)

    for sp_ in sp_resolution, sp_resolution0:
        sp_.plot(t_axis*1e15, resolution*1e15, label=label)

for sp_ in sp_resolution, sp_resolution0:
    sp_.set_ylim(0, 10)

for sp_ in sp_beamsize, sp_resolution, sp_resolution0:
    sp_.legend()






if args.savedir:
    ms.saveall(args.savedir, empty_suptitle=True, ending='.pdf', trim=False, wspace=0.7)

if not args.noshow:
    plt.show()



# Resolution = beamsize / (dx/dt)

# Copy from 045h script

