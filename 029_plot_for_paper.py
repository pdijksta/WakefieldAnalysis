#here are the filenames for the figures for the passive, which I hope could/would be used for the paper:
#
#
#1) 38 fs:
#
#Passive_data_20201003T233852.mat (October 3, 2020)
#
#
##fileN='Passive_data_20201003T233852.mat' #lasing off/on
#
###fileLoff='Passive_data_20201003T231958.mat' - (scan of the offsets)
#
#
#
#
#2) 25 fs:
#
#Passive_data_20201004T172425.mat (October 4th, 2020)
#
#
#fileN='Passive_data_20201004T172425.mat' #scan different offsets oct04
#
##fileN='Passive_data_20201004T163828.mat'
#
#
#
#3)The tilted beam for shorter pulses (March 1st):
#
#https://elog-gfa.psi.ch/SwissFEL+commissioning/13854
#
#
#Look for the datafiles between:
#
#21:48 and 22:30
#
#
#Here is the description from the shift summary:
#
#    We are imposing tilt in Y with skew quad in BC2 (S10BCO2-MQSK350) {one in BC1 gives similar results} to generate shorter pulses:
#        we get visible effect: only few slices are lasing   22611
#        we save data for different tilt strenghts (post-undulator LPS images and spectra)
#        for best tilt (tilt4) we saved data for different lasing slices by changing correctors: SARUN02-MCRY080, SARUN03-MCRY080 (more effective)
#
#
#
#Best,
#
#Alex.

import os
import mat73
import copy; copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt
#from scipy.constants import c

import elegant_matrix
import tracking
import gaussfit
import misc

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 0.03
profile_cutoff = 0.00
len_profile = int(5e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = [500e-9, 500e-9]
n_particles = int(100e3)
n_streaker = 1
flip_measured = False
#sig_t_range = np.arange(20, 40.01, 2)*1e-15

mean_struct2 = 472e-6 # see 026_script
gap2_correcting_summand = -40e-6
sig_t_range = np.arange(20, 40.01, 2)*1e-15
gaps = [10e-3, 10e-3+gap2_correcting_summand]

# According to Alex, use data from these days:
# https://elog-gfa.psi.ch/SwissFEL+commissioning/16450 (4th October 2020)
# https://elog-gfa.psi.ch/SwissFEL+commissioning/16442 (3rd October 2020)

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


blmeas_1 = dirname1+'129833611_bunch_length_meas.h5'
energy_eV = 4491892915.7690735
profile_meas = tracking.profile_from_blmeas(blmeas_1, tt_halfrange, charge, energy_eV, subtract_min=True)
profile_meas.reshape(len_profile)
profile_meas2 = tracking.profile_from_blmeas(blmeas_1, tt_halfrange, charge, energy_eV, subtract_min=True, zero_crossing=2)
profile_meas2.reshape(len_profile)
if flip_measured:
    profile_meas.flipx()
else:
    profile_meas2.flipx()



file0 = dirname1 + 'Passive_data_20201003T231958.mat'

file_ = dirname1 + 'Passive_data_20201003T233852.mat'

file25 = dirname2 + 'Passive_data_20201004T172425.mat'

dict0 = mat73.loadmat(file0)

dict25 = mat73.loadmat(file25)

x_axis = dict0['x_axis']*1e-6
y_axis = dict0['y_axis']*1e-6

if np.diff(x_axis)[0] < 0:
    x_axis = x_axis[::-1]
    invert_x = True
else:
    invert_x = False

if np.diff(y_axis)[0] < 0:
    y_axis = y_axis[::-1]
    invert_y = True
else:
    invert_y = False

def get_screen_from_image(image):
    projX = np.sum(image, axis=0)
    if invert_x:
        screen = tracking.ScreenDistribution((-x_axis[::-1]).copy(), (projX[::-1]).copy())
    else:
        screen = tracking.ScreenDistribution(x_axis.copy(), projX.copy())
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen

images0 = dict0['Image'][-1]
all_mean, all_mean2 = [], []
for n_img in range(len(images0)):
    screen = get_screen_from_image(dict0['Image'][-1][n_img].T)
    xx, yy = screen._xx, screen._yy
    gf = gaussfit.GaussFit(xx, yy)
    all_mean.append(gf.mean)
    all_mean2.append(xx[np.argmax(yy)])

mean0 = np.mean(all_mean)
mean02 = np.mean(all_mean2)

timestamp0 = misc.get_timestamp(os.path.basename(file0))
tracker0 = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp0, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile)
r120 = tracker0.calcR12()[n_streaker]

bp_test = tracking.get_gaussian_profile(40e-15, tt_halfrange, len_profile, charge, tracker0.energy_eV)
screen_sim = tracker0.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']
all_emittances = []
for n_img in range(len(images0)):
    screen_meas = get_screen_from_image(dict0['Image'][-1][n_img].T)
    emittance_fit = misc.fit_nat_beamsize(screen_meas, screen_sim, n_emittances[0])
    all_emittances.append(emittance_fit)

new_emittance = np.mean(all_emittances)
n_emittances[0] = new_emittance

dict_ = mat73.loadmat(file_)
timestamp  = misc.get_timestamp(os.path.basename(file_))
tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile)

beam_offsets = [0., -(dict_['value']*1e-3 - mean_struct2)]
distance_um = (gaps[n_streaker]/2. - beam_offsets[n_streaker])*1e6

fig_paper = ms.figure('Comparison plots')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp_profile_comp = subplot(sp_ctr, title='Temporal reconstruction', xlabel='t [fs]', ylabel='Intensity (arb. units)')
sp_ctr += 1
profile_meas.plot_standard(sp_profile_comp, norm=True, color='black', label='TDC', center_max=True)

ny, nx = 2, 4
subplot = ms.subplot_factory(ny, nx)
sp_ctr = np.inf

all_profiles, all_screens = [], []
for n_image in range(len(dict_['Image'])):
    image = dict_['Image'][n_image].T
    screen = get_screen_from_image(image)
    screen.crop()
    screen._xx = screen._xx - mean0

    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True)
    best_screen = gauss_dict['reconstructed_screen']
    best_screen.cutoff(1e-3)
    best_screen.crop()
    best_profile = gauss_dict['reconstructed_profile']
    if n_image == 0:
        screen00 = screen
        bp00 = best_profile
        best_screen00 = best_screen
    best_gauss = gauss_dict['best_gauss']

    if sp_ctr > (ny*nx):
        ms.figure('All reconstructions Distance %i' % distance_um)
        sp_ctr = 1

    if n_image % 5 == 0:
        sp_profile = subplot(sp_ctr, title='Reconstructions')
        sp_ctr += 1
        sp_screen = subplot(sp_ctr, title='Screens')
        sp_ctr += 1
        profile_meas.plot_standard(sp_profile, color='black', label='Measured', norm=True, center_max=True)
    color = screen.plot_standard(sp_screen, label=n_image)[0].get_color()
    best_screen.plot_standard(sp_screen, color=color, ls='--')
    best_profile.plot_standard(sp_profile, label=n_image, norm=True, center_max=True)
    sp_profile.legend()
    sp_screen.legend()

    all_profiles.append(best_profile)

# Averaging the reconstructed profiles
all_profiles_time, all_profiles_current = [], []
for profile in all_profiles:
    all_profiles_time.append(profile.time - profile.time[np.argmax(profile.current)])
new_time = np.linspace(min(x.min() for x in all_profiles_time), max(x.max() for x in all_profiles_time), len_profile)
for tt, profile in zip(all_profiles_time, all_profiles):
    all_profiles_current.append(np.interp(new_time, tt, profile.current, left=0, right=0))
all_profiles_current = np.array(all_profiles_current)
mean_profile = np.mean(all_profiles_current, axis=0)
std_profile = np.std(all_profiles_current, axis=0)
average_profile = tracking.BeamProfile(new_time, mean_profile, energy_eV, charge)
average_profile.plot_standard(sp_profile_comp, label='Reconstructed', norm=True, center_max=True)


ms.figure('Test averaging')
sp = plt.subplot(1,1,1)
for yy in all_profiles_current:
    sp.plot(new_time, yy, lw=0.5)

to_plot = [
        ('Average', new_time, mean_profile, 'black', 3),
        ('+1 STD', new_time, mean_profile+std_profile, 'black', 1),
        ('-1 STD', new_time, mean_profile-std_profile, 'black', 1),
        ]

integral = np.trapz(mean_profile, new_time)
for pm, ctr, color in [(profile_meas, 1, 'red'), (profile_meas2, 2, 'green')]:
    factor = integral/np.trapz(pm.current, pm.time)
    t_meas = pm.time-pm.time[np.argmax(pm.current)]
    i_meas = pm.current*factor

    to_plot.append(('TDC %i' % ctr, t_meas, i_meas, color, 3))


for label, tt, profile, color, lw in to_plot:
    gf = gaussfit.GaussFit(tt, profile)
    width_fs = gf.sigma*1e15
    if label is None:
        label = ''
    label = (label + ' %i fs' % width_fs).strip()
    sp.plot(tt, profile, color=color, lw=lw, label=label)

sp.legend(title='Gaussian fit $\sigma$')

plt.show()
