import re
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
screen_cutoff = 0.02
profile_cutoff = 0.00
len_profile = int(5e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = [500e-9, 500e-9]
n_particles = int(100e3)
n_streaker = 1
flip_measured = False

beta_x0 = 4.968
alpha_x0 = -0.563
beta_y0 = 16.807
alpha_y0 = 1.782

optics0a = [beta_x0, beta_y0, alpha_x0, alpha_y0]
#optics0a = 'default'
optics0b = [50, 50, 0, 0]

mean_struct2 = 472e-6 # see 026_script
gap2_correcting_summand = -40e-6
sig_t_range = np.arange(20, 40.01, 1)*1e-15

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

# Try for largest offset
timestamp = get_timestamp(file_)

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

ms.figure('Comparison')
subplot = ms.subplot_factory(1,2)
sp_ctr = 1


sp_profile_comp = subplot(sp_ctr, title='Profiles', xlabel='t [fs]', ylabel='Intensity (arb. units)', scix=True, sciy=True)
sp_ctr += 1

profile_meas.plot_standard(sp_profile_comp, norm=True, label='TDC', center_max=True)


for n_optics, optics0 in enumerate([optics0a, optics0b]):
    tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, optics0=optics0)
    r12 = tracker.calcR12()[n_streaker]

    bp_test = tracking.get_gaussian_profile(40e-15, tt_halfrange, len_profile, charge, energy_eV)
    forward0 = tracker.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])
    screen_sim = forward0['screen']

    emittances_fit = []
    for n_image, image in enumerate(dict_['Image'][-1]):
        screen_meas = tracking.ScreenDistribution(x_axis, image.T.sum(axis=0))
        emittance_fit = misc.fit_nat_beamsize(screen_meas, screen_sim, n_emittances[0],print_=n_image==0)
        emittances_fit.append(emittance_fit)

        #if n_image == 0:
        #    ms.figure('Optics %i' % n_optics)
        #    sp = plt.subplot(1,1,1)
        #    sp.plot(screen_meas.x-screen_meas.gaussfit.mean, screen_meas.intensity/screen_meas.intensity.max(), label='Meas')
        #    sp.plot(screen_sim.x, screen_sim.intensity/screen_sim.intensity.max(), label='Sim')
        #    sp.legend()

    emittances_fit = np.array(emittances_fit)
    mean_emittance = emittances_fit.mean()

    n_emittances = [mean_emittance, 500e-9]
    tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, optics0=optics0)
    image = dict_['Image'][0][0].T
    meas_screen = get_screen_from_image(image)
    meas_screen.crop()
    meas_screen._xx = meas_screen._xx - mean0


    test = tracker.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])
    test_beam = test['screen']
    beamsize = test_beam.gaussfit.sigma
    beta = np.sqrt(beamsize**2/(mean_emittance/(energy_eV/511e3)))
    real_beamsize = meas_screen.gaussfit.sigma
    print('n_optics, beta, mean_emittance, beamsize, real_beamsize')
    print(n_optics, beta, mean_emittance, beamsize, real_beamsize)

    gaps = [10e-3, 10e-3+gap2_correcting_summand]
    offset_arr = dict_['value']*1e-3 - mean_struct2
    beam_offsets = [0., -offset_arr[0]]

    distance_um = (gaps[n_streaker]/2. - abs(beam_offsets[n_streaker]))*1e6

    gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge)
    print(n_optics, gauss_dict['gauss_sigma'])

    best_screen = gauss_dict['reconstructed_screen']
    best_profile = bp000 = gauss_dict['reconstructed_profile']
    best_gauss = gauss_dict['best_gauss']

    optics_label = r'$\beta$=%.1f $\alpha$=%.1f' % (optics0[0], optics0[1])
    best_profile.plot_standard(sp_profile_comp, label=optics_label, norm=True, center_max=True)

    ms.figure('Reconstruction Distance %i $\mu$m N_optics %i' % (distance_um, n_optics))
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_screen = subplot(sp_ctr, title='Screens', xlabel='x [mm]', ylabel='Intensity (arb. units)', scix=True, sciy=True)
    sp_ctr += 1

    sp_profile = subplot(sp_ctr, title='Beam profile', xlabel='t [fs]', ylabel='Current (arb. units)')
    sp_ctr += 1

    profile_meas.plot_standard(sp_profile, label='TDC', color='black', norm=True, center_max=True)
    best_profile.plot_standard(sp_profile, norm=True, label='Use self', center_max=True)
    best_gauss.plot_standard(sp_profile, norm=True, label='Best Gauss', center_max=True)

    meas_screen.plot_standard(sp_screen, label='Measurement', color='black')
    best_screen.plot_standard(sp_screen, label='Reconstruction')

    sp_profile.legend()
    sp_screen.legend()


sp_profile_comp.legend()



plt.show()



