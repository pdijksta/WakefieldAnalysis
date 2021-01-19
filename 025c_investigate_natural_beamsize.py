import re
import socket
import numpy as np
import matplotlib.pyplot as plt
import mat73

import elegant_matrix
import tracking
import gaussfit

import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant')

sig_t = 40e-15 # for Gaussian beam
tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 0.03
profile_cutoff = 0.00
len_profile = 5e3
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 0e-6
n_emittances = (1800e-9, 200e-9)
n_particles = int(100e3)
n_streaker = 1
flip_measured = False

hostname = socket.gethostname()
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

#image = dict_['Image'][0][0].T
image0 = dict_['Image'][-1][0].T
#meas_screen = get_screen_from_image(image, invert=True)
meas_screen0 = get_screen_from_image(image0, invert=True)


timestamp = get_timestamp(file_)


blmeas_1 = dirname1+'129833611_bunch_length_meas.h5'
energy_eV = 4491892915.7690735
profile_meas = tracking.profile_from_blmeas(blmeas_1, tt_halfrange, charge, energy_eV, subtract_min=True)


tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile)
forward_dict = tracker.matrix_forward(profile_meas, [10e-3, 10e-3], [0., 0.])
forward_dict_ele = tracker.elegant_forward(profile_meas, [10e-3, 10e-3], [0., 0.])
beam_forward = forward_dict['beam0_at_screen']

image_sim, xedges, yedges = np.histogram2d(beam_forward[0], beam_forward[2], bins=(200, 100), normed=True)

fig = ms.figure('Compare images')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_img = subplot(sp_ctr, title='Measured', grid=False)
sp_ctr += 1
sp_img.imshow(image0, aspect='auto', extent=(x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]))

sp_img = subplot(sp_ctr, title='Simulated', grid=False)
sp_ctr += 1
sp_img.imshow(image_sim, aspect='auto', extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]))

sp_proj = subplot(sp_ctr, title='Projections X')
sp_ctr += 1

sp_projY = subplot(sp_ctr, title='Projections Y')
sp_ctr += 1

histx, xedges = np.histogram(beam_forward[0], bins=200)
histy, yedges = np.histogram(beam_forward[2], bins=200)

for arr, arry, axis, axisy, label in [(image0.sum(axis=0), image0.sum(axis=1), x_axis, y_axis, 'Measured'), (histx, histy, xedges[1:], yedges[1:], 'Simulated')]:

    for sp, arr_, ax in [(sp_proj, arr, axis), (sp_projY, arry, axisy)]:
        gf = gaussfit.GaussFit(ax, arr_)
        sigma = gf.sigma
        x_max = ax[np.argmax(arr_)]
        sp.plot(ax-x_max, arr_/arr_.max(), label=label +' %i' % (sigma*1e6))

sp_proj.legend()
sp_projY.legend()


sp_ctr = np.inf
n_proj = np.inf
ny, nx = 3, 3
subplot = ms.subplot_factory(ny, nx)


images0 = dict_['Image'][-1]

all_means = []

for n_image, image in enumerate(images0):
    image = image.T
    projX = np.sum(image, axis=0)
    gf = gaussfit.GaussFit(x_axis, projX)

    if sp_ctr > nx*ny:
        ms.figure('All unstreaked images')
        sp_ctr = 1
    if n_proj > 5:
        sp = subplot(sp_ctr, title='Projections', scix=True)
        sp_ctr += 1
        n_proj = 0

    color = sp.plot(x_axis, projX)[0].get_color()
    sp.axvline(gf.mean, color=color, ls='--')
    sp.plot(gf.xx, gf.reconstruction, ls='--', color=color)
    all_means.append(gf.mean)
    n_proj += 1
    sp.set_xlim(-0.001, 0.001)

all_means = np.array(all_means)

sp_all = subplot(sp_ctr, title='All means')
sp_ctr += 1
sp_all.hist(all_means*1e6)



plt.show()



