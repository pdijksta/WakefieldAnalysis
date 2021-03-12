"""
No SwissFEL / PSI specific imports in this file.
Should handle analysis, saving and reloading of data.
"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import tracking
import myplotstyle as ms
import h5_storage
from datetime import datetime

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

    def current_profile_rec_gauss(self, kwargs, do_plot, plot_handles=None):
        """
        kwargs are getting prepared for input to the find_best_gauss procedure.
        plot_handles need to be matplotlib axes for (screen, profile, opt)
        """
        kwargs['meas_screen']._xx = kwargs['meas_screen']._xx - self.input_data['screen_x0']
        kwargs['beam_offsets'] = np.array(kwargs['beam_offsets'])
        kwargs['beam_offsets'] -= np.array(self.input_data['streaker_means'])
        kwargs['beam_offsets'] *= -1
        kwargs['meas_screen'].cutoff(self.input_data['tracker_kwargs']['screen_cutoff'])
        kwargs['meas_screen'].crop()

        # Only allow one streaker at the moment
        for n in (0,1):
            if n != kwargs['n_streaker']:
                kwargs['beam_offsets'][n] = 0

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
                ms.figure('Optimization')
                sp_ctr = 1
                subplot = ms.subplot_factory(2,2)
                sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Intensity (arb. units)')
                sp_ctr += 1
                meas_screen.plot_standard(sp_screen, label='Original', color='black', lw=3)
                sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
                sp_ctr += 1
                sp_opt = subplot(sp_ctr, title='Optimization')
                sp_ctr += 1
            else:
                sp_screen, sp_profile, sp_opt = plot_handles

            for opt_ctr, (screen, profile, value, sigma) in enumerate(zip(opt_func_screens, opt_func_profiles, opt_func_values[:,1], opt_func_sigmas)):
                screen.plot_standard(sp_screen, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value))
                profile.plot_standard(sp_profile, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value), center='Gauss')

            best_screen.plot_standard(sp_screen, color='red', lw=3, label='Final')
            best_profile.plot_standard(sp_profile, color='red', lw=3, label='Final', center='Gauss')

            sp_screen.legend()
            sp_profile.legend()

            sp_opt.scatter(opt_func_sigmas*1e15, opt_func_values[:,1])
            plt.show()


    def save_data(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        date = datetime.now()
        filename = os.path.join(save_path, date.strftime('%Y%m%d_%H%M%S_PassiveReconstruction.h5'))
        import pdb; pdb.set_trace()
        save_dict = {
                'input': self.input_data,
                'gaussian_reconstruction': self.gauss_dict,
                }
        h5_storage.saveH5Recursive(filename, save_dict)
        return filename

def load_reconstruction(filename):
    pass

if __name__ == '__main__':
    plt.close('all')
    import pickle
    with open('/tmp/tmp_gauss_dict.pkl', 'rb') as f:
        gauss_dict, tracker_args, kwargs = pickle.load(f)

    #tracker = tracking.Tracker(**tracker_args)
    #gauss_dict = tracker.find_best_gauss(**kwargs)

    best_profile = gauss_dict['reconstructed_profile']
    best_screen = gauss_dict['reconstructed_screen']
    opt_func_values = gauss_dict['opt_func_values']
    opt_func_screens = gauss_dict['opt_func_screens']
    opt_func_profiles = gauss_dict['opt_func_profiles']
    opt_func_sigmas = np.array(gauss_dict['opt_func_sigmas'])
    meas_screen = gauss_dict['meas_screen']

    ms.figure('Optimization')
    sp_ctr = 1
    subplot = ms.subplot_factory(2,2)
    sp_screen = subplot(sp_ctr, title='Screen', xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    meas_screen.plot_standard(sp_screen, label='Original', color='black', lw=3)
    sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
    sp_ctr += 1
    sp_opt = subplot(sp_ctr, title='Optimization')
    sp_ctr += 1

    for opt_ctr, (screen, profile, value, sigma) in enumerate(zip(opt_func_screens, opt_func_profiles, opt_func_values[:,1], opt_func_sigmas)):
        screen.plot_standard(sp_screen, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value))
        profile.plot_standard(sp_profile, label='%i: %.1f fs %.3e' % (opt_ctr, sigma*1e15, value), center='Gauss')

    gauss_dict['reconstructed_screen'].plot_standard(sp_screen, color='red', lw=3, label='Final')
    gauss_dict['reconstructed_profile'].plot_standard(sp_profile, color='red', lw=3, label='Final', center='Gauss')

    sp_screen.legend()
    sp_profile.legend()

    sp_opt.scatter(opt_func_sigmas*1e15, opt_func_values[:,1])
    plt.show()

