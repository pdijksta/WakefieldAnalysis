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

try:
    import tracking
    import myplotstyle as ms
    import h5_storage
    import elegant_matrix
    import gaussfit
    import misc2 as misc
except ImportError:
    from . import tracking
    from . import myplotstyle as ms
    from . import h5_storage
    from . import elegant_matrix
    from . import gaussfit
    from . import misc2 as misc

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

    def current_profile_rec_gauss(self, kwargs, do_plot, plot_handles=None, debug=False):

        kwargs_save = copy.deepcopy(kwargs)
        del kwargs_save['meas_screen']
        kwargs_save['meas_screen_x'] = kwargs['meas_screen'].x
        kwargs_save['meas_screen_intensity'] = kwargs['meas_screen'].intensity
        self.input_data['gaussian_reconstruction'] = kwargs_save

        if debug:
            import matplotlib
            matplotlib.use('TKAgg')

            ms.figure('Debug find_best_gauss')
            subplot = ms.subplot_factory(1,1)
            sp = subplot(1)
            meas_screen = kwargs['meas_screen']
            meas_screen.plot_standard(sp)
            plt.show()
            import pdb; pdb.set_trace()

        gauss_dict = self.tracker.find_best_gauss(**kwargs)
        #import pickle
        #with open('/tmp/tmp_gauss_dict.pkl', 'wb') as f:
        #    pickle.dump((gauss_dict, self.tracker_args, kwargs), f)

        self.gauss_dict = gauss_dict

        print('Do plotting')
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
        filename = os.path.join(save_path, date.strftime('%Y_%m_%d-%H_%M_%S_PassiveReconstruction.h5'))
        #gauss_dict = self.gauss_dict
        #gauss_dict2 = copy.deepcopy(gauss_dict)
        #gauss_dict2['reconstructed_screen_x'] = gauss_dict['reconstructed_screen'].x
        #gauss_dict2['reconstructed_screen_intensity'] = gauss_dict['reconstructed_screen'].intensity
        save_dict = {
                'input': self.input_data,
                'gaussian_reconstruction': self.gauss_dict,
                }
        #import pdb; pdb.set_trace()
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


def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, const, semigap):
    wall0, wall1 = -semigap, semigap
    c1 = np.abs((offsets-streaker_offset-wall0))**(-order)
    c2 = np.abs((offsets-streaker_offset-wall1))**(-order)
    return const + (c1 - c2)*strength

def analyze_streaker_calibration(filename_or_dict, do_plot=True, plot_handles=None):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))

    if 'raw_data' in data_dict:
        data_dict = data_dict['raw_data']
    result_dict = data_dict['pyscan_result']

    images = result_dict['image'].astype(float).squeeze()
    proj_x = np.sum(images, axis=-2).squeeze()
    x_axis = result_dict['x_axis']
    offsets = data_dict['streaker_offsets'].squeeze()
    n_images = int(data_dict['n_images'])

    centroids = np.zeros([len(offsets), n_images])
    if n_images > 1:
        for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
            centroids[n_o,n_i] = np.sum(proj_x[n_o,n_i]*x_axis) / np.sum(proj_x[n_o,n_i])
        centroid_mean = np.mean(centroids, axis=1)
        centroid_std = np.std(centroids, axis=1)
    elif n_images == 1:
        for n_o in range(len(offsets)):
            centroids[n_o] = np.sum(proj_x[n_o]*x_axis) / np.sum(proj_x[n_o])
        centroid_mean = centroids.squeeze()
        centroid_std = np.ones_like(centroid_mean)*1e-10

    streaker = data_dict['streaker']
    semigap = data_dict['meta_data'][streaker+':GAP']/2.*1e-3

    wall0, wall1 = -semigap, semigap

    where0 = np.argwhere(offsets == 0).squeeze()
    const0 = centroid_mean[where0]
    delta_offset0 = (offsets.min() + offsets.max())/2
    order0 = 3

    s01 = (centroid_mean[0] - const0) / (np.abs((offsets[0]-wall0))**(-order0) - np.abs((offsets[0]-wall1))**(-order0))
    s02 = (centroid_mean[-1] - const0) / (np.abs((offsets[-1]-wall0))**(-order0) - np.abs((offsets[-1]-wall1))**(-order0))
    strength0 = (s01 + s02) / 2

    p0 = [delta_offset0, strength0, order0, const0]

    def fit_func(*args):
        return streaker_calibration_fit_func(*args, semigap)

    try:
        p_opt, p_cov = curve_fit(fit_func, offsets, centroid_mean, p0, sigma=centroid_std)
    except RuntimeError:
        print('Streaker calibration did not converge')
        p_opt = p0
    xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
    reconstruction = fit_func(xx_fit, *p_opt)
    initial_guess = fit_func(xx_fit, *p0)
    streaker_offset = p_opt[0]

    meta_data = {
            'p_opt': p_opt,
            'p0': p0,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'offsets': offsets,
            'semigap': semigap,
            'streaker_offset': streaker_offset,
            'reconstruction': reconstruction,
            }

    output = {
            'raw_data': data_dict,
            'meta_data': meta_data,
            }

    if not do_plot:
        return output

    if plot_handles is None:
        fig, (sp_center, ) = streaker_calibration_figure()
    else:
        (sp_center, ) = plot_handles
    screen = data_dict['screen']
    sp_center.set_title(screen)

    xx_plot = (offsets - streaker_offset)*1e3
    xx_plot_fit = (xx_fit - streaker_offset)*1e3
    sp_center.errorbar(xx_plot, centroid_mean, yerr=centroid_std, label='Data')
    sp_center.plot(xx_plot_fit, reconstruction, label='Fit')
    sp_center.plot(xx_plot_fit, initial_guess, label='Guess')
    sp_center.legend()

    return output

def analyze_screen_calibration(filename_or_dict, do_plot=True, plot_handles=None):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))

    if 'pyscan_result' in data_dict:
        screen_data = data_dict['pyscan_result']
    else:
        screen_data = data_dict

    x_axis = screen_data['x_axis']
    if 'projx' in screen_data:
        projx = screen_data['projx']
    else:
        images = screen_data['image'].astype(float).squeeze()
        projx = images.sum(axis=-2)

    all_mean = []
    all_std = []
    for proj in projx:
        gf = gaussfit.GaussFit(x_axis, proj)
        all_mean.append(gf.mean)
        all_std.append(gf.sigma)

    index_median = np.argsort(all_mean)[len(all_mean)//2]
    projx_median = projx[index_median]
    beamsize = np.mean(all_std)

    x0 = gaussfit.GaussFit(x_axis, projx_median).mean
    output = {
            'raw_data': data_dict,
            'x0': x0,
            'beamsize': beamsize,
            }

    if not do_plot:
        return output

    if plot_handles is None:
        fig, (sp_proj,) = screen_calibration_figure()
    else:
        (sp_proj, ) = plot_handles

    for proj in projx:
        sp_proj.plot(x_axis*1e3, proj)
    sp_proj.plot(x_axis*1e3, projx_median, lw=3)
    sp_proj.axvline(x0*1e3)

    return output

def screen_calibration_figure():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.35)
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 1)

    sp_proj = subplot(sp_ctr, xlabel='x [mm]', ylabel='Intensity (arb. units)', sciy=True)
    sp_ctr += 1
    clear_screen_calibration(sp_proj)
    return fig, (sp_proj, )

def clear_screen_calibration(sp_proj):
    for sp, title, xlabel, ylabel in [
            (sp_proj, 'Unstreaked beam', 'x [mm]', 'Intensity (arb. units)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def reconstruction_figure():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    sp_ctr = 1
    subplot = ms.subplot_factory(2,2)
    sp_screen = subplot(sp_ctr)
    sp_ctr += 1
    sp_profile = subplot(sp_ctr)
    sp_ctr += 1
    sp_opt = subplot(sp_ctr)
    sp_ctr += 1
    clear_reconstruction(sp_screen, sp_profile, sp_opt)

    return fig, (sp_screen, sp_profile, sp_opt)

def clear_reconstruction(sp_screen, sp_profile, sp_opt):
    for sp, title, xlabel, ylabel in [
            (sp_screen, 'Screen', 'x [mm]', 'Intensity (arb. units)'),
            (sp_profile, 'Profile', 't [fs]', 'Current [kA]'),
            (sp_opt, 'Optimization', 'Gaussian $\sigma$ [fs]', 'Opt value'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def streaker_calibration_figure():
    fig = plt.figure()
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 1)

    sp_center = subplot(sp_ctr, xlabel='Streaker center [mm]', ylabel='Beam X centroid [mm]')
    sp_ctr += 1
    clear_streaker_calibration(sp_center)
    return fig, (sp_center, )

def clear_streaker_calibration(sp_center):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Screen center', 'Streaker center [mm]', 'Beam X centroid [mm]')
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def reconstruct_lasing(file_on, file_off, key_on, key_off, screen_center, file_current, r12, gap, beam_offset, struct_length):
    input_dict = {
            'file_on': file_on,
            'file_off': file_off,
            'key_on': key_on,
            'key_off': key_off,
            'file_current': file_current,
            'r12': r12,
            'gap': gap,
            'beam_offset': beam_offset,
            'struct_length': struct_length,
            }

    dict_on0 = h5_storage.loadH5Recursive(file_on)
    if key_on is not None:
        dict_on = dict_on0[key_on]
    else:
        dict_on = dict_on0

    dict_off0 = h5_storage.loadH5Recursive(file_off)
    if key_off is not None:
        dict_off = dict_off0[key_off]
    else:
        dict_off = dict_off0

    images0 = dict_off['image'].astype(np.float64)
    if 'x_axis_m' in dict_off:
        x_axis0 = dict_off['x_axis_m'].astype(np.float64)
    else:
        x_axis0 = dict_off['x_axis'].astype(np.float64)*1e-6
    projx0 = images0.sum(axis=-2)
    proj_median_screen = misc.get_median(projx0, output='proj')
    median_screen_off = misc.proj_to_screen(proj_median_screen, x_axis0, True, screen_center)

    current_dict = h5_storage.loadH5Recursive(file_current)
    wake_profile = current_dict['gaussian_reconstruction']['reconstructed_profile']
    tt, xx = wake_profile.get_x_t(gap, beam_offset, struct_length, r12)

    import pdb; pdb.set_trace()



if __name__ == '__main__':
    plt.close('all')

    filename = '/tmp/2021_03_16-20_22_26_Screen_data_SARBD02-DSCR050.h5'
    dict_ = h5_storage.loadH5Recursive(filename)

    plt.show()

