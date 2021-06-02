import copy
import numpy as np
import matplotlib.pyplot as plt
from socket import gethostname
from scipy.optimize import curve_fit

from h5_storage import loadH5Recursive
import config
import tracking
import image_and_profile as iap
import myplotstyle as ms

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')
blmeas_file = data_dir+'119325494_bunch_length_meas.h5'

tracker_kwargs = config.get_default_tracker_settings()
magnet_data_file = data_dir + '2021_05_18-21_27_16_Lasing_True_SARBD02-DSCR050.h5'
magnet_data = loadH5Recursive(magnet_data_file)['meta_data_begin']

tracker_kwargs['magnet_file'] = magnet_data
tracker_kwargs['profile_cutoff'] = 2e-2

tracker = tracking.Tracker(**tracker_kwargs)

distances = np.arange(250, 500.01, 25)*1e-6
offset0 = 0
tt_halfrange = 200e-15
charge = 200e-12

gaps = [10e-3, 10e-3]

beam_profile = iap.profile_from_blmeas(blmeas_file, tt_halfrange, charge, tracker.energy_eV, True, 1)
beam_profile.reshape(tracker.len_screen)
beam_profile.cutoff2(tracker_kwargs['profile_cutoff'])
beam_profile.crop()
beam_profile.reshape(tracker.len_screen)

gauss_sigma = beam_profile.rms()

gauss_profile = iap.get_gaussian_profile(gauss_sigma, tt_halfrange, tracker_kwargs['len_screen'], charge, tracker.energy_eV, cutoff=tracker_kwargs['profile_cutoff'])

ms.figure('Forward propagated jaw distance scaling', figsize=(26, 13))
subplot = ms.subplot_factory(2,3)
sp_ctr = 1
sp_profile = subplot(sp_ctr, xlabel='t (fs)', ylabel='I (kA)', title='Current profile')
sp_ctr += 1
sp_centroids = subplot(sp_ctr, xlabel='d ($\mu$m)', ylabel='$\Delta$x (mm)', title='Centroid deflection')
sp_ctr += 1
sp_rms = subplot(sp_ctr, xlabel='d ($\mu$m)', ylabel=r'$\sqrt{\langle x^2\rangle}$ (mm)', title='Beam size')
sp_ctr += 1

def fit_func_centroid(distances, order, strength):
    return strength * distances**-order

def fit_func_rms(distances, order, strength):
    return np.sqrt(beamsize0**2 + strength**2 * distances**(-2*order))

fit_xx = np.linspace(distances.min(), distances.max(), 100)
order0 = 2.7

for profile, profile_label in [(gauss_profile, 'Gauss'), (beam_profile, 'TDC')]:

    sp_screen = subplot(sp_ctr, xlabel='x (mm)', ylabel='Screen intensity (arb. units)', title='Screen projection %s' % profile_label)
    sp_ctr += 1

    for scale_factor, ls, marker in [(0.5, 'dotted', '.'), (1, None, 'o'), (1.5, '--', 'x')]:
        scaled_profile = copy.deepcopy(profile)
        scaled_profile.scale_xx(scale_factor, keep_range=False)

        _label = '%s %i fs' % (profile_label, round(scaled_profile.rms()*1e15))
        scaled_profile.plot_standard(sp_profile, label=_label, ls=ls)

        centroids, rms = [], []

        for distance1 in distances:
            distance_um = distance1*1e6
            beam_offsets = [offset0, gaps[1]/2-distance1]
            forward_dict = tracker.matrix_forward(scaled_profile, gaps, beam_offsets)
            screen = forward_dict['screen']
            screen.plot_standard(sp_screen, label='%i' % distance_um, ls=ls)
            centroids.append(screen.mean())
            rms.append(screen.rms())

        screen0 = tracker.matrix_forward(scaled_profile, gaps, [0, 0])['screen']
        beamsize0 = screen0.rms()

        centroids = np.array(centroids)
        rms = np.array(rms)

        for y_arr, sp, fit_func in [(centroids, sp_centroids, fit_func_centroid), (rms, sp_rms, fit_func_rms)]:
            p0 = [order0, y_arr.max()*distances.min()**order0]
            p_opt, p_cov = curve_fit(fit_func, distances, y_arr, p0)
            fit_order = p_opt[0]
            fit_yy = fit_func(fit_xx, *p_opt)

            _label = '%s %.2f %.2f' % (profile_label, scale_factor, fit_order)
            color = sp.plot(distances*1e6, y_arr*1e3, label=_label, marker=marker, ls='None')[0].get_color()
            sp.plot(fit_xx*1e6, fit_yy*1e3, color=color)

sp_profile.legend(title='rms')
sp_profile.set_xlim(-100, 100)

for sp_ in sp_centroids, sp_rms:
    sp_.legend(title='Scale factor, fit scaling')

ms.saveall('/tmp/055_fit_model')

plt.show()

