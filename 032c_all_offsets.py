import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname
from h5_storage import loadH5Recursive
import gaussfit
from scipy.optimize import curve_fit

import tracking
import elegant_matrix


import myplotstyle as ms

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

len_profile = int(5e3)
charge = -200e-12
energy_eV = 4.4918e9
struct_lengths = [1., 1.]
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
invert_offset = True

quad_wake = True
override_quad_beamsize = False
quad_x_beamsize = [0, 10e-6]



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

blmeas38 = dirname1+'129833611_bunch_length_meas.h5'


file0 = dirname1 + 'Passive_data_20201003T231958.mat'

dict0 = loadH5Recursive(file0+'.h5')

subtract_min = True

def get_screen_from_proj(projX, x_axis, invert_x):
    if invert_x:
        xx, yy = (x_axis[::-1]).copy(), (projX[::-1]).copy()
    else:
        xx, yy = x_axis.copy(), projX.copy()
    if subtract_min:
        yy -= yy.min()
    screen = tracking.ScreenDistribution(xx, yy)
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen

x_axis = dict0['x_axis']*1e-6
invert_x0 = (np.diff(x_axis)[0] < 0)
projx = dict0['projx']
projx0 = dict0['projx'][-1]
all_mean = []
for proj in projx0:
    screen = get_screen_from_proj(proj, x_axis, invert_x0)
    xx, yy = screen._xx, screen._yy
    gf = gaussfit.GaussFit(xx, yy)
    all_mean.append(gf.mean)

mean0 = np.mean(all_mean)


profile_meas = tracking.profile_from_blmeas(blmeas38, tt_halfrange, charge, energy_eV)

offset_arr = dict0['value']*1e-3 - mean_offset
if invert_offset:
    offset_arr *= -1


ms.figure('Summary')
subplot = ms.subplot_factory(1, 1)
sp_summary = subplot(1, xlabel='x [mm]', ylabel='Intensity (arb. units)')

ms.figure('Centroid')
sp_centroid = subplot(1, xlabel='Offset [mm]', ylabel='Centroid')
centroid_list = []
centroid_list_sig = []
centroid_list_sim = []


offsets = []
for n_offset, offset in enumerate(offset_arr[:1]):
    offsets.append(offset)

    all_sigma = []
    all_screens = []
    all_centroid = []
    for n_screen in range(len(projx[n_offset])):
        screen = get_screen_from_proj(projx[n_offset][n_screen], x_axis, invert_x0)
        screen._xx = screen._xx - mean0
        all_sigma.append(screen.gaussfit.sigma)
        all_screens.append(screen)
        all_centroid.append(np.sum(screen.x * screen.intensity) / np.sum(screen.intensity))
    index = np.argsort(all_sigma)[len(all_sigma)//2]
    print(offset, index)
    avg_screen = all_screens[index]

    #avg_screen._xx = avg_screen._xx - mean0

    #centroid_list.append(np.sum(avg_screen.x * avg_screen.intensity) / np.sum(avg_screen.intensity))
    centroid_list.append(np.mean(all_centroid))
    centroid_list_sig.append(np.std(all_centroid))


    ms.figure('Forward propagate offset %.2f mm' % (offset*1e3))
    subplot = ms.subplot_factory(1,2)
    sp_ctr = 1

    sp_profile = subplot(sp_ctr, title='Current profile', xlabel='time [fs]', ylabel='I (arb. units)')
    sp_ctr += 1
    profile_meas.plot_standard(sp_profile)
    sp_profile.set_xlim(-100, 100)

    sp_forward = subplot(sp_ctr, title='Screen distribution', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    avg_screen.plot_standard(sp_forward, label='Measured', lw=3, color='black')
    color = avg_screen.plot_standard(sp_summary, label='%i $\mu$m' % ((5e-3 - offset)*1e6))[0].get_color()

    tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, bp_smoothen=bp_smoothen, quad_wake=quad_wake)

    tracker.override_quad_beamsize = override_quad_beamsize
    tracker.quad_x_beamsize = quad_x_beamsize

    beam_offsets = [0, offset]

    forward_dict = tracker.matrix_forward(profile_meas, gaps, beam_offsets)
    screen = forward_dict['screen_no_smoothen']
    screen.smoothen(smoothen)

    bs = forward_dict['bs_at_streaker'][1]

    screen.crop()
    screen.plot_standard(sp_forward, label='%i $\mu$m' % (bs*1e6))
    screen.plot_standard(sp_summary, color=color, ls='--')

    centroid_list_sim.append(np.sum(screen.x * screen.intensity) / np.sum(screen.intensity))


    #sp_forward.legend(title='RMS beamsize at streaker')
    sp_forward.set_xlim(-0.3, 1.5)

centroids = np.array(centroid_list)
centroids_sig = np.array(centroid_list_sig)
centroids_sim = np.array(centroid_list_sim)
offsets = np.array(offsets)


order = 3
def fit_func(offsets, wall, scale):
    return 1/(offsets-wall)**order * scale

for centroid_arr, err, label in [(centroids, centroids_sig*1e3, 'Measured'), (centroids_sim, None, 'Simulated')]:

    color = sp_centroid.errorbar(offsets*1e3, centroid_arr*1e3, label=label, yerr=err)[0].get_color()
    try:
        fit, _ = curve_fit(fit_func, offsets, centroid_arr, (gaps[1]/2., 1e-14))
        reconstruction = fit_func(offsets, *fit)

        sp_centroid.plot(offsets*1e3, reconstruction*1e3, color=color, ls='--', label='%.4f' % (fit[0]*1e3))
    except (RuntimeError, TypeError):
        print('runtimerror')
        pass

sp_centroid.legend()

sp_summary.legend(title='Measured')
sp_summary.set_xlim(-0.3, 1.5)

ms.saveall('/tmp/032c')

plt.show()

