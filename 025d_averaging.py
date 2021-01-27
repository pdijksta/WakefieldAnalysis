import mat73
import copy; copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt
from scipy.constants import c

import elegant_matrix
import tracking
import gaussfit
import misc
from misc import re_file, get_timestamp

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 0.03
profile_cutoff = 0.00
len_profile = int(5e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = (2200e-9, 500e-9)
n_particles = int(100e3)
n_streaker = 1
flip_measured = False
optics0 = [10, 0, 10, 0]

mean_struct2 = 472e-6 # see 026_script
gap2_correcting_summand = -40e-6

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

files1 = [
        #'Passive_data_20201003T231812.mat',
        'Passive_data_20201003T231958.mat',
        'Passive_data_20201003T233852.mat',]

files2 = [
        'Passive_data_20201004T161118.mat',
        'Passive_data_20201004T172425.mat',
        'Passive_data_20201004T223859.mat',
        'Passive_data_20201004T163828.mat',
        'Passive_data_20201004T221502.mat',
        #'Passive_money_20201004T012247.mat',
        ]
#blmeas_1 = dirname1+'Bunch_length_meas_2020-10-03_15-43-29.h5'
blmeas_1 = dirname1+'129833611_bunch_length_meas.h5'
blmeas_2 = dirname2+'129858802_bunch_length_meas.h5'

#blmeas_1 = blmeas_1


energy_eV = 4491892915.7690735
profile_meas = tracking.profile_from_blmeas(blmeas_1, tt_halfrange, charge, energy_eV, subtract_min=True)
#profile_meas.cutoff(1e-3)
profile_meas.reshape(len_profile)
if flip_measured:
    profile_meas.flipx()

good_files = [
        'Passive_data_20201003T231958.mat',
        'Passive_data_20201003T233852.mat',
        'Passive_data_20201004T161118.mat',
        'Passive_data_20201004T172425.mat',
        'Passive_data_20201004T223859.mat',
        ]


def get_file(f):
    day = re_file.match(f).group(3)
    if day == '03':
        return dirname1+f
    elif day == '04':
        return dirname2+f
    else:
        raise ValueError

file_ = good_files[0]
dict_ = mat73.loadmat(get_file(file_))

x_axis = dict_['x_axis']*1e-6
y_axis = dict_['y_axis']*1e-6

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

def get_img(i, j):
    img = dict_['Image'][i][j].T
    if invert_x:
        img = img[:,::-1]
    if invert_y:
        img = img[::-1,:]
    return img




sp_ctr = np.inf

offset_arr = dict_['value']*1e-3 - mean_struct2
if offset_arr.size == 1:
    offset_arr = np.array([offset_arr])
for o_index, offset in enumerate(offset_arr):

    if sp_ctr > 8:
        fig = ms.figure('Images')
        subplot = ms.subplot_factory(2,4)
        sp_ctr = 1
    image = dict_['Image'][o_index][0].T

    if np.diff(x_axis)[0] < 0:
        image = image[:,::-1]

    if np.diff(y_axis)[0] < 0:
        image = image[::-1,:]

    sp_img = subplot(sp_ctr, grid=False, title='Offset %.2f mm' % (offset*1e3), scix=True, sciy=True)
    sp_ctr += 1

    sp_img.imshow(image, aspect='auto', extent=(x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]))

    sp_proj = subplot(sp_ctr, title='Projections', scix=True, sciy=True)
    sp_ctr += 1

    sp_proj.plot(x_axis, image.sum(axis=0), label='X')
    sp_proj.plot(y_axis, image.sum(axis=1), label='Y')
    sp_proj.legend()


# Try for largest offset
timestamp = get_timestamp(file_)
tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, optics0=optics0)
r12 = tracker.calcR12()[n_streaker]

bp_test = tracking.get_gaussian_profile(40e-15, tt_halfrange, len_profile, charge, energy_eV)
screen_sim = tracker.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']

emittances_fit = []
for n_image, image in enumerate(dict_['Image'][-1]):
    screen_meas = tracking.ScreenDistribution(x_axis, image.T.sum(axis=0))
    emittance_fit = misc.fit_nat_beamsize(screen_meas, screen_sim, n_emittances[0])
    emittances_fit.append(emittance_fit)

emittances_fit = np.array(emittances_fit)
mean_emittance = emittances_fit.mean()

n_emittances = [mean_emittance, 500e-9]
tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, optics0=optics0)





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

images0 = dict_['Image'][-1]
all_mean, all_mean2 = [], []
for n_img in range(len(images0)):
    #img0 = get_img(-1, n_img)
    #if False:
    #    xx = -x_axis[::-1]
    #    yy = img0.sum(axis=0)[::-1]
    #else:
    #    xx = x_axis
    #    yy = img0.sum(axis=0)
    screen = get_screen_from_image(dict_['Image'][-1][n_img].T)
    xx, yy = screen._xx, screen._yy
    gf = gaussfit.GaussFit(xx, yy)
    all_mean.append(gf.mean)
    all_mean2.append(xx[np.argmax(yy)])

mean0 = np.mean(all_mean)
mean02 = np.mean(all_mean2)



ms.figure('Reconstructed profiles')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile_rec = subplot(sp_ctr, title='Beam profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

image0 = dict_['Image'][-1][0].T
meas_screen0 = get_screen_from_image(image0)
x0_old = meas_screen0.x[np.argmax(meas_screen0.intensity)]
x0 = mean0
meas_screen0._xx = meas_screen0._xx - x0

for offset_index in range(1):

    sig_t_range = np.arange(20, 40.01, 2)*1e-15
    image = dict_['Image'][offset_index][0].T
    meas_screen = get_screen_from_image(image)
    beam_offsets = [0., -offset_arr[offset_index]]

    gaps = [10e-3, 10e-3+gap2_correcting_summand]
    distance_um = (gaps[n_streaker]/2. - abs(beam_offsets[n_streaker]))*1e6

    meas_screen._xx = meas_screen._xx - x0
    meas_screen.crop()

    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True)

    best_screen = gauss_dict['reconstructed_screen']
    best_profile = bp000 = gauss_dict['reconstructed_profile']
    best_gauss = gauss_dict['best_gauss']

    ms.figure('Reconstruction Distance %i $\mu$m' % distance_um)
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_screen = subplot(sp_ctr, title='Screens', xlabel='x [mm]', ylabel='Intensity (arb. units)', scix=True, sciy=True)
    sp_ctr += 1

    sp_profile = subplot(sp_ctr, title='Beam profile', xlabel='t [fs]', ylabel='Current (arb. units)')
    sp_ctr += 1

    profile_meas.plot_standard(sp_profile, label='TDC', color='black', norm=True)
    best_profile.plot_standard(sp_profile, norm=True, label='Use self')
    color_rec = best_profile.plot_standard(sp_profile_rec, norm=True, label='%i' % distance_um)[0].get_color()
    best_gauss.plot_standard(sp_profile, norm=True, label='Best Gauss')

    sp_wake = subplot(sp_ctr, title='Wake effect', xlabel='t [fs]', ylabel='$\Delta$ x', scix=True, sciy=True)
    sp_ctr += 1

    baf_tdc = tracker.back_and_forward(meas_screen, profile_meas, gaps, beam_offsets, n_streaker)
    bp_back_tdc = baf_tdc['beam_profile']
    screen_tdc = baf_tdc['screen']
    #bp_back_tdc = tracker.track_backward2(meas_screen, profile_meas, gaps, beam_offsets, n_streaker)
    bp_back_tdc.plot_standard(sp_profile, norm=True, label='Use measured')
    bp_back_tdc.plot_standard(sp_profile_rec, norm=True, ls='--')
    mask = np.logical_and(bp_back_tdc.time > -1e-13, bp_back_tdc.time < 1e-13)
    bp_back_tdc2 = tracking.BeamProfile(bp_back_tdc.time[mask], bp_back_tdc.current[mask], bp_back_tdc.energy_eV, bp_back_tdc.charge)

    sp_profile.legend()

    for bp, label in [(best_profile, 'Use self'), (best_gauss, 'Best Gauss'), (bp_back_tdc2, 'Use measured'), (profile_meas, 'TDC measured')]:
        wake_dict = bp.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker])
        wake = wake_dict['dipole']['wake_potential']
        wake_t = wake_dict['input']['charge_xx']/c

        sp_wake.plot(wake_t, wake/tracker.energy_eV*r12, label=label)

    sp_wake.legend()

    meas_screen0.plot_standard(sp_screen, label='Measured 0')
    meas_screen.plot_standard(sp_screen, label='Measured')
    best_screen.plot_standard(sp_screen, label='Reconstructed')
    screen_tdc.plot_standard(sp_screen, label='Reconstructed use TDC')
    sp_screen.legend()

    profile_meas.plot_standard(sp_profile_rec, label='TDC', color='black', norm=True)



    subplot = ms.subplot_factory(2,2)
    sp_ctr = np.inf

    all_profiles = []
    for n_image in range(len(dict_['Image'][offset_index])):
        image = dict_['Image'][offset_index][n_image].T
        screen = get_screen_from_image(image)
        screen.crop()
        screen._xx = screen._xx - x0

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

        if sp_ctr > 4:
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
    for profile in all_profiles:
        all_profiles_current.append(np.interp(new_time, profile.time, profile.current, left=0, right=0))
    all_profiles_current = np.array(all_profiles_current)
    average_profile = tracking.BeamProfile(new_time, np.mean(all_profiles_current, axis=0), energy_eV, charge)
    average_profile.plot_standard(sp_profile_rec, label='Average', norm=True)

sp_profile_rec.legend()


#import pickle
#with open('./025d.pkl', 'wb') as f:
#    pickle.dump({
#        'meas_screen': screen00,
#        'best_profile': bp00,
#        'best_screen': best_screen00},f)




plt.show()

