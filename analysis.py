"""
No SwissFEL / PSI specific imports in this file.
Should handle analysis, saving and reloading of data.
"""
import itertools
import os
from datetime import datetime
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tracking
import myplotstyle as ms
import h5_storage
import elegant_matrix

class Reconstruction:
    def __init__(self):
        self.input_data = {}

    def add_tracker(self, tracker_args):
        self.tracker = tracking.Tracker(**tracker_args)
        self.input_data['tracker_kwargs'] = tracker_args

    def add_streaker_means(self, streaker_means):
        self.input_data['streaker_means'] = streaker_means

    def add_screen_x0(self, screen_x0):
        self.input_data['screen_x0'] = screen_x0

        #self.init_dict = {
        #        'tracker_args': tracker_args,
        #        'streaker_means': streaker_means,
        #        'screen_x0': screen_x0,
        #        }

    def prepare_rec_gauss_args(self, kwargs):
        """
        kwargs are getting prepared for input to the find_best_gauss procedure.
        """
        kwargs = copy.deepcopy(kwargs)
        kwargs['meas_screen']._xx = kwargs['meas_screen']._xx - self.input_data['screen_x0']
        streaker_offsets = np.array(kwargs['streaker_offsets'])
        kwargs['beam_offsets'] = -(streaker_offsets - np.array(self.input_data['streaker_means']))
        del kwargs['streaker_offsets']
        kwargs['meas_screen'].cutoff(self.input_data['tracker_kwargs']['screen_cutoff'])
        kwargs['meas_screen'].crop()

        # Only allow one streaker at the moment
        for n in (0,1):
            if n != kwargs['n_streaker']:
                kwargs['beam_offsets'][n] = 0

        return kwargs

    def current_profile_rec_gauss(self, kwargs, do_plot, plot_handles=None):

        kwargs_save = copy.deepcopy(kwargs)
        del kwargs_save['meas_screen']
        kwargs_save['meas_screen_x'] = kwargs['meas_screen'].x
        kwargs_save['meas_screen_intensity'] = kwargs['meas_screen'].intensity
        self.input_data['gaussian_reconstruction'] = kwargs_save
        gauss_dict = self.tracker.find_best_gauss(**kwargs)
        #import pickle
        #with open('/tmp/tmp_gauss_dict.pkl', 'wb') as f:
        #    pickle.dump((gauss_dict, self.tracker_args, kwargs), f)

        self.gauss_dict = gauss_dict

        if do_plot:
            best_profile = gauss_dict['reconstructed_profile']
            best_screen = gauss_dict['reconstructed_screen']
            opt_func_values = gauss_dict['opt_func_values']
            opt_func_screens = gauss_dict['opt_func_screens']
            opt_func_profiles = gauss_dict['opt_func_profiles']
            opt_func_sigmas = np.array(gauss_dict['opt_func_sigmas'])
            meas_screen = gauss_dict['meas_screen']

            if plot_handles is None:
                fig, (sp_screen, sp_profile, sp_opt) = reconstruction_figure()
                plt.suptitle('Optimization')
            else:
                sp_screen, sp_profile, sp_opt = plot_handles

            meas_screen.plot_standard(sp_screen, color='black', lw=3)

            for opt_ctr, (screen, profile, value, sigma) in enumerate(zip(opt_func_screens, opt_func_profiles, opt_func_values[:,1], opt_func_sigmas)):
                screen.plot_standard(sp_screen, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value))
                profile.plot_standard(sp_profile, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value), center='Gauss')

            best_screen.plot_standard(sp_screen, color='red', lw=3, label='Final')
            best_profile.plot_standard(sp_profile, color='red', lw=3, label='Final', center='Gauss')

            sp_screen.legend()
            sp_profile.legend()

            sp_opt.scatter(opt_func_sigmas*1e15, opt_func_values[:,1])
            if plot_handles is None:
                plt.show()


    def save_data(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        date = datetime.now()
        filename = os.path.join(save_path, date.strftime('%Y%m%d_%H%M%S_PassiveReconstruction.h5'))
        save_dict = {
                'input': self.input_data,
                'gaussian_reconstruction': self.gauss_dict,
                }
        h5_storage.saveH5Recursive(filename, save_dict)
        return filename

def load_reconstruction(filename, tmp_dir, plot_handles=None):
    elegant_matrix.set_tmp_dir(tmp_dir)
    saved_dict = h5_storage.loadH5Recursive(filename)
    analysis_obj = Reconstruction()
    analysis_obj.add_tracker(saved_dict['input']['tracker_kwargs'])
    rec_kwargs = saved_dict['input']['gaussian_reconstruction']
    rec_kwargs['meas_screen'] = tracking.ScreenDistribution(rec_kwargs['meas_screen_x'], rec_kwargs['meas_screen_intensity'])
    del rec_kwargs['meas_screen_x']
    del rec_kwargs['meas_screen_intensity']
    analysis_obj.current_profile_rec_gauss(rec_kwargs, True, plot_handles)
    return analysis_obj


def streaker_calibration_fit_func(offsets, delta_offset, strength, order, const, semigap):
    wall0, wall1 = -semigap, semigap
    c1 = np.abs((offsets-delta_offset-wall0))**(-order)
    c2 = np.abs((offsets-delta_offset-wall1))**(-order)
    return const + (c1 - c2)*strength

def analyze_streaker_calibration(filename_or_dict, do_plot=True, plot_handles=None):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))
    result_dict = data_dict['pyscan_result']

    images = result_dict['image'].astype(float)
    proj_x = np.sum(images, axis=1)
    x_axis = result_dict['x_axis']*1e-6
    offsets = data_dict['streaker_offsets']
    n_images = int(data_dict['n_images'])

    centroids = np.zeros([len(offsets), n_images])
    for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
        centroids[n_o,n_i] = np.sum(proj_x[n_o,n_i]*x_axis) / np.sum(proj_x[n_o,n_i])

    centroid_mean = np.mean(centroids, axis=1)
    centroid_std = np.std(centroids, axis=1)

    streaker = data_dict['streaker']
    semigap = data_dict['meta_data'][streaker+':GAP']/2.

    wall0, wall1 = -semigap, semigap

    where0 = np.argwhere(offsets == 0).squeeze()
    const0 = centroid_mean[where0]
    delta_offset0 = 0
    order0 = 3

    s01 = (centroid_mean[0] - const0) / (np.abs((offsets[0]-wall0))**(-order0) - np.abs((offsets[0]-wall1))**(-order0))
    s02 = (centroid_mean[-1] - const0) / (np.abs((offsets[-1]-wall0))**(-order0) - np.abs((offsets[-1]-wall1))**(-order0))
    strength0 = (s01 + s02) / 2

    p0 = [delta_offset0, strength0, order0, const0]

    def fit_func(*args):
        return streaker_calibration_fit_func(*args, semigap)

    p_opt, p_cov = curve_fit(fit_func, offsets, centroid_mean, p0, sigma=centroid_std)
    reconstruction = fit_func(*p_opt)
    delta_offset = p_opt[0]

    meta_data = {
            'p_opt': p_opt,
            'p0': p0,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'offsets': offsets,
            'semigap': semigap,
            'delta_offset': delta_offset,
            'reconstruction': reconstruction,
            }

    output = {
            'raw_data': data_dict,
            'meta_data': meta_data,
            }

    if not do_plot:
        return output

    if plot_handles is None:
        screen = data_dict['screen']
        fig, (sp_center, ) = streaker_calibration_figure(screen)
    else:
        (sp_center, ) = plot_handles

    xx_plot = offsets - delta_offset
    sp_center.errorbar(xx_plot, centroid_mean, yerr=centroid_std, label='Data')
    sp_center.plot(xx_plot, reconstruction, label='Fit')
    sp_center.legend()

    return output

def reconstruction_figure():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.35)
    sp_ctr = 1
    subplot = ms.subplot_factory(2,2)
    sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
    sp_ctr += 1
    sp_opt = subplot(sp_ctr, title='Optimization')
    sp_ctr += 1
    return fig, (sp_screen, sp_profile, sp_opt)

def streaker_calibration_figure(screen):

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.35)
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 1)

    sp_center = subplot(sp_ctr, title=screen, xlabel='Center (mm)', ylabel='Streaker offset (mm)', grid=False)
    sp_ctr += 1
    return fig, (sp_center, )

if __name__ == '__main__':
    dirname = '/home/work/data_2020-10-03/'
    file_ = dirname+'Passive_alignment_20201003T221023.mat'

    import scipy.io as sio
    dict_ = sio.loadmat(file_)


    #analyze_streaker_calibration(file_



