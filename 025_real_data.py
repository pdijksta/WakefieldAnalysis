import re
import mat73
import copy; copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt
from scipy.constants import c

import elegant_matrix
import tracking

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

sig_t = 40e-15 # for Gaussian beam
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

#file_ = dirname1 + files1[-1]
#dict_ = loadH5Recursive(file_)



#import h5py
#good_files = []
#for f in files1+files2:
##for f in files2:
#    if f in files1:
#        file_ = dirname1 + f
#    elif f in files2:
#        file_ = dirname2 + f
#    print(f)
#    if not os.path.isfile(file_):
#        print(file_, 'not exists')
#        continue
#    try:
#        with h5py.File(file_, 'r') as f2:
#            print(list(f2.keys()))
#            if 'value' in f2:
#                print(np.array(f2['value']))
#                good_files.append(file_)
#            else:
#                print('No value in %s' % file_)
#    except:
#        continue

good_files = [
        'Passive_data_20201003T231958.mat',
        'Passive_data_20201003T233852.mat',
        'Passive_data_20201004T161118.mat',
        'Passive_data_20201004T172425.mat',
        'Passive_data_20201004T223859.mat',
        ]

re_file = re.compile('Passive_data_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2}).mat')
def get_timestamp(filename):
    match = re_file.match(filename)
    args = [int(x) for x in match.groups()]
    if match is None:
        print(filename)
        raise ValueError
    return elegant_matrix.get_timestamp(*args)


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

image0 = dict_['Image'][0][0].T
if np.diff(x_axis)[0] < 0:
    x_axis = x_axis[::-1]
    image0 = image0[:,::-1]

if np.diff(y_axis)[0] < 0:
    y_axis = y_axis[::-1]
    image0 = image0[::-1,:]


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
tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile)
r12 = tracker.calcR12()[n_streaker]

def get_screen_from_image(image, invert=False):
    projX = np.sum(image, axis=0)
    if invert:
        screen = tracking.ScreenDistribution(-x_axis[::-1], projX[::-1])
    else:
        screen = tracking.ScreenDistribution(x_axis, projX)
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen

ms.figure('Reconstructed profiles')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_profile_rec = subplot(sp_ctr, title='Beam profile', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

offset_index = 0
#for offset_index in range(len(dict_['Image'])-1):
for offset_index in range(1):

    sig_t_range = np.arange(20, 40.01, 2)*1e-15
    image = dict_['Image'][offset_index][0].T
    image0 = dict_['Image'][-1][0].T
    meas_screen = get_screen_from_image(image, invert=True)
    meas_screen0 = get_screen_from_image(image0, invert=True)
    beam_offsets = [0., -offset_arr[offset_index]]

    gaps = [10e-3, 10e-3+gap2_correcting_summand]
    distance = (gaps[n_streaker]/2. - abs(beam_offsets[n_streaker]))*1e6

    x0 = meas_screen.x[np.argmax(meas_screen0.intensity)]
    meas_screen0._xx = meas_screen0._xx - x0
    meas_screen._xx = meas_screen._xx - x0

    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge)

    best_screen = gauss_dict['reconstructed_screen']
    best_profile = gauss_dict['reconstructed_profile']
    best_gauss = gauss_dict['best_gauss']
    print(gauss_dict['gauss_sigma'])

    ms.figure('Reconstruction Distance %i $\mu$m' % distance)
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_screen = subplot(sp_ctr, title='Screens', xlabel='x [mm]', ylabel='Intensity (arb. units)', scix=True, sciy=True)
    sp_ctr += 1

    sp_profile = subplot(sp_ctr, title='Beam profile', xlabel='t [fs]', ylabel='Current (arb. units)')
    sp_ctr += 1

    profile_meas.plot_standard(sp_profile, label='TDC', color='black', norm=True)
    best_profile.plot_standard(sp_profile, norm=True, label='Use self')
    color_rec = best_profile.plot_standard(sp_profile_rec, norm=True, label='%i' % distance)[0].get_color()
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

sp_profile_rec.legend()

plt.show()

