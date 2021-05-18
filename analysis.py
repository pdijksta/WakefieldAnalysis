"""
No SwissFEL / PSI specific imports in this file.
"""
import itertools
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import tracking
    import myplotstyle as ms
    import h5_storage
    import gaussfit
    import misc2 as misc
    import image_and_profile as iap
    import lasing
    import config
except ImportError:
    from . import tracking
    from . import myplotstyle as ms
    from . import h5_storage
    from . import gaussfit
    from . import misc2 as misc
    from . import image_and_profile as iap
    from . import lasing
    from . import config

def plt_show():
    plt.pause(.1)
    plt.draw()
    plt.show(block=False)

class Reconstruction:
    def __init__(self, screen_x0, streaker_means):
        self.input_data = {
                'screen_x0': screen_x0,
                'streaker_means': streaker_means,
                }

    def add_tracker(self, tracker_args):
        self.tracker = tracking.Tracker(**tracker_args)
        self.input_data['tracker_kwargs'] = tracker_args

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

    def current_profile_rec_gauss(self, kwargs, do_plot, plot_handles=None, blmeas_file=None, debug=False):

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
            plt_show()
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

            if blmeas_file is not None:
                blmeas_profiles = []
                for zero_crossing in (1, 2):
                    try:
                        blmeas_profile = iap.profile_from_blmeas(blmeas_file, kwargs['tt_halfrange'], kwargs['charge'], self.tracker.energy_eV, True, zero_crossing)
                        blmeas_profile.cutoff2(5e-2)
                        blmeas_profile.crop()
                        blmeas_profile.reshape(int(1e3))
                        blmeas_profiles.append(blmeas_profile)
                    except KeyError as e:
                        print(e)
                        print('No zero crossing %i in %s' % (zero_crossing, blmeas_file))

                for blmeas_profile, ls, zero_crossing in zip(blmeas_profiles, ['--', 'dotted'], [1, 2]):
                    blmeas_profile.plot_standard(sp_profile, ls=ls, color='black', label='Blmeas %i' % zero_crossing)


            sp_screen.legend()
            sp_profile.legend()

            yy_opt = opt_func_values[:,1]
            sp_opt.scatter(opt_func_sigmas*1e15, yy_opt)
            sp_opt.set_ylim(0,1.1*yy_opt.max())
            if plot_handles is None:
                plt_show()

def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, const, semigap):
    wall0, wall1 = -semigap, semigap
    c1 = np.abs((offsets-streaker_offset-wall0))**(-order)
    c2 = np.abs((offsets-streaker_offset-wall1))**(-order)
    return const + (c1 - c2)*strength

def analyze_streaker_calibration(filename_or_dict, do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))

    result_dict = data_dict['pyscan_result']

    if 'image' in result_dict:
        images = result_dict['image'].astype(float).squeeze()
        proj_x = np.sum(images, axis=-2).squeeze()
    elif 'projx' in result_dict:
        proj_x = result_dict['projx']

    if 'x_axis_m' in result_dict:
        x_axis = result_dict['x_axis_m']
    else:
        x_axis = result_dict['x_axis']

    offsets = data_dict['streaker_offsets'].squeeze()
    n_images = int(data_dict['n_images'])

    centroids = np.zeros([len(offsets), n_images])

    plot_list = []
    if n_images > 1:
        for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
            proj = proj_x[n_o,n_i]
            proj -= np.median(proj)
            proj[np.abs(proj)< np.abs(proj).max()*0.02] = 0
            if n_i == 0:
                plot_list.append(proj)
            centroids[n_o,n_i] = np.sum(proj*x_axis) / np.sum(proj)
        centroid_mean = np.mean(centroids, axis=1)
        centroid_std = np.std(centroids, axis=1)
    elif n_images == 1:
        for n_o in range(len(offsets)):
            proj = proj_x[n_o]
            proj -= np.median(proj)
            proj[np.abs(proj)< np.abs(proj).max()*0.05] = 0
            centroids[n_o] = np.sum(proj_x[n_o]*x_axis) / np.sum(proj_x[n_o])
            plot_list.append(proj)
        centroid_mean = centroids.squeeze()
        centroid_std = None

    streaker = data_dict['streaker']
    semigap = data_dict['meta_data_end'][streaker+':GAP']/2.*1e-3


    if force_screen_center is None:
        where0 = np.argwhere(offsets == 0).squeeze()
        const0 = centroid_mean[where0]
    else:
        const0 = force_screen_center
    delta_offset0 = (offsets.min() + offsets.max())/2
    order0 = 3
    wall0, wall1 = -semigap+delta_offset0, semigap+delta_offset0

    s01 = (centroid_mean[0] - const0) / (np.abs((offsets[0]-wall0))**(-order0) - np.abs((offsets[0]-wall1))**(-order0))
    s02 = (centroid_mean[-1] - const0) / (np.abs((offsets[-1]-wall0))**(-order0) - np.abs((offsets[-1]-wall1))**(-order0))
    strength0 = (s01 + s02) / 2

    p0 = [delta_offset0, strength0]
    if fit_order:
        p0.append(order0)

    def fit_func(*args):
        if fit_order:
            return streaker_calibration_fit_func(*args, const0, semigap)
        else:
            return streaker_calibration_fit_func(*args, order0, const0, semigap)

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
            'centroids': centroids,
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

    if do_plot:

        if plot_handles is None:
            fig, (sp_center, sp_proj) = streaker_calibration_figure()
        else:
            (sp_center, sp_proj) = plot_handles
        screen = data_dict['screen']
        sp_center.set_title(screen)

        xx_plot = (offsets - streaker_offset)*1e3
        xx_plot_fit = (xx_fit - streaker_offset)*1e3
        sp_center.errorbar(xx_plot, (centroid_mean-const0)*1e3, yerr=centroid_std*1e3, label='Data', ls='None', marker='o')
        sp_center.plot(xx_plot_fit, (reconstruction-const0)*1e3, label='Fit')
        sp_center.plot(xx_plot_fit, (initial_guess-const0)*1e3, label='Guess')
        sp_center.legend()

        for proj, offset in zip(plot_list, offsets):
            sp_proj.plot((x_axis-const0)*1e3, proj, label='%.2f mm' % (offset*1e3))
        sp_proj.legend()

    return output

def analyze_screen_calibration(filename_or_dict, do_plot=True, plot_handles=None):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))

    screen_data = data_dict['pyscan_result']
    if 'x_axis_m' in screen_data:
        x_axis = screen_data['x_axis_m']
    else:
        print(screen_data['x_axis'].shape)
        x_axis = screen_data['x_axis'][0]*1e-6

    assert len(x_axis.squeeze().shape) == 1

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

    projx_median -= projx_median.min()
    x0 = gaussfit.GaussFit(x_axis, projx_median).mean
    #x02 = np.sum(projx_median*x_axis) / np.sum(projx_median)
    #x0 = x02
    #import pdb; pdb.set_trace()
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
        sp_proj.plot(x_axis*1e3, proj-proj.min())
    sp_proj.plot(x_axis*1e3, projx_median-projx_median.min(), lw=3)
    sp_proj.axvline(x0*1e3)

    return output

def screen_calibration_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Screen center calibration')
    fig.subplots_adjust(hspace=0.35)
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 1)

    sp_proj = subplot(sp_ctr, xlabel='x (mm)', ylabel='Intensity (arb. units)', sciy=True)
    sp_ctr += 1
    clear_screen_calibration(sp_proj)
    return fig, (sp_proj, )

def clear_screen_calibration(sp_proj):
    for sp, title, xlabel, ylabel in [
            (sp_proj, 'Unstreaked beam', 'x (mm)', 'Intensity (arb. units)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def reconstruction_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Current reconstruction')
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
            (sp_screen, 'Screen', 'x (mm)', 'Intensity (arb. units)'),
            (sp_profile, 'Profile', 't (fs)', 'Current (kA)'),
            (sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def streaker_calibration_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Streaker center calibration')
    sp_ctr = 1
    subplot = ms.subplot_factory(1, 2)

    sp_center = subplot(sp_ctr)
    sp_ctr += 1
    sp_proj = subplot(sp_ctr)
    clear_streaker_calibration(sp_center, sp_proj)
    return fig, (sp_center, sp_proj)

def clear_streaker_calibration(sp_center, sp_proj):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Screen center', 'Streaker center [mm]', 'Beam X centroid [mm]'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def lasing_figures():

    output = []

    fig = plt.figure()
    fig.canvas.set_window_title('Lasing reconstruction')
    subplot = ms.subplot_factory(3,3, grid=False)
    sp_ctr = 1

    sp_profile = subplot(sp_ctr)
    sp_ctr += 1

    sp_wake = subplot(sp_ctr)
    sp_ctr += 1

    sp_off = subplot(sp_ctr)
    sp_ctr += 1

    sp_on = subplot(sp_ctr)
    sp_ctr += 1

    sp_off_cut = subplot(sp_ctr)
    sp_ctr += 1

    sp_on_cut = subplot(sp_ctr)
    sp_ctr += 1

    sp_off_tE = subplot(sp_ctr)
    sp_ctr += 1

    sp_on_tE = subplot(sp_ctr)
    sp_ctr += 1

    output.append((fig, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)


    fig = plt.figure()
    subplot = ms.subplot_factory(2,2, grid=False)
    sp_ctr = 1

    sp_power = subplot(sp_ctr)
    sp_ctr += 1

    sp_current = subplot(sp_ctr)
    sp_ctr += 1
    output.append((fig, (sp_power, sp_current)))

    clear_lasing(output)

    return output

def clear_lasing(plot_handles):
    (_, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)) = plot_handles[0]
    (_, (sp_power, sp_current)) = plot_handles[1]

    for sp, title, xlabel, ylabel in [
            (sp_profile, 'Current profile', 't (fs)', 'I (kA)'),
            (sp_wake, 'Wake', 't (fs)', 'x (mm)'),
            (sp_off, 'Lasing off', 'x (mm)', 'y (mm)'),
            (sp_on, 'Lasing on', 'x (mm)', 'y (mm)'),
            (sp_off_cut, 'Lasing off', 'x (mm)', 'y (mm)'),
            (sp_on_cut, 'Lasing on', 'x (mm)', 'y (mm)'),
            (sp_off_tE, 'Lasing off', 't (fs)', '$\Delta$ E (MeV)'),
            (sp_on_tE, 'Lasing off', 't (fs)', '$\Delta$ E (MeV)'),
            (sp_power, 'Power', 't (fs)', 'P (GW)'),
            (sp_current, 'Current', 't (fs)', 'I (arb. units)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)

def reconstruct_lasing(file_on, file_off, screen_center, structure_center, structure_length, file_current, r12, disp, energy_eV, charge, streaker, plot_handles, pulse_energy):
    input_dict = {
            'file_on': file_on,
            'file_off': file_off,
            'screen_center': screen_center,
            'structure_center': structure_center,
            'structure_length': structure_length,
            'file_current': file_current,
            'r12': r12,
            'disp': disp,
            'energy_eV': energy_eV,
            'charge': charge,
            'streaker': streaker,
            #'gap': gap,
            #'beam_offset': beam_offset,
            #'struct_length': struct_length,
            }

    dict_on = h5_storage.loadH5Recursive(file_on)
    dict_on_p = dict_on['pyscan_result']
    dict_on_m = dict_on['meta_data']

    dict_off = h5_storage.loadH5Recursive(file_off)
    dict_off_p = dict_off['pyscan_result']
    dict_off_m = dict_off['meta_data']

    gaps, beam_offsets = [], []
    for dict_ in dict_off_m, dict_on_m:
        gaps.append(dict_[streaker+':GAP']*1e-3)
        beam_offsets.append(-(dict_[streaker+':CENTER']*1e-3-structure_center))

    if abs(gaps[0] - gaps[1]) > 1e-6:
        print('Gaps not the same!', gaps)
    if abs(beam_offsets[0] - beam_offsets[1]) > 1e-6:
        print('Beam offsets not the same!', beam_offsets)
    gap = np.mean(gaps)
    beam_offset = np.mean(beam_offsets)

    if pulse_energy is None:
        try:
            pulse_energy = dict_on_m[config.gas_monitor_pvs['Aramis']]*1e-6
        except KeyError:
            print('No pulse energy found! Use 100 uJ')
            pulse_energy = 100e-6

    images0 = dict_off_p['image'].astype(np.float64)
    images0_on = dict_on_p['image'].astype(np.float64)

    if 'x_axis_m' in dict_off_p:
        x_axis0 = dict_off_p['x_axis_m'].astype(np.float64)
        y_axis0 = dict_off_p['y_axis_m'].astype(np.float64)
    else:
        x_axis0 = dict_off_p['x_axis'].astype(np.float64)
        y_axis0 = dict_off_p['y_axis'].astype(np.float64)

    projx0 = images0.sum(axis=-2)
    median_index = misc.get_median(projx0, output='index')
    median_image_off = iap.Image(images0[median_index], x_axis0, y_axis0, x_offset=screen_center)

    projx0_on = images0_on.sum(axis=-2)
    median_index = misc.get_median(projx0_on, output='index')
    median_image_on = iap.Image(images0_on[median_index], x_axis0, y_axis0, x_offset=screen_center)

    # TODO
    n_slices = 50
    len_profile = 2000

    current_dict = h5_storage.loadH5Recursive(file_current)
    wake_profile_dict = current_dict['gaussian_reconstruction']['reconstructed_profile']
    wake_profile = iap.BeamProfile.from_dict(wake_profile_dict)
    wake_profile.cutoff2(0.2)
    wake_profile.crop()
    wake_profile.reshape(len_profile)

    wake_t, wake_x = wake_profile.get_x_t(gap, beam_offset, structure_length, r12)

    lasing_dict = lasing.obtain_lasing(median_image_off, median_image_on, n_slices, wake_x, wake_t, len_profile, disp, energy_eV, charge, pulse_energy=pulse_energy, debug=False)

    if plot_handles is None:
        plot_handles = lasing_figures()

    (fig, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)) = plot_handles[0]
    (fig, (sp_power, sp_current)) = plot_handles[1]

    slice_time = lasing_dict['slice_time']
    all_slice_dict = lasing_dict['all_slice_dict']
    power_from_Eloss = lasing_dict['power_Eloss']
    power_from_Espread = lasing_dict['power_Espread']

    sp_current.plot(slice_time*1e15, all_slice_dict['Lasing_off']['slice_current'], label='Off')
    sp_current.plot(slice_time*1e15, all_slice_dict['Lasing_on']['slice_current'], label='On')

    sp_power.plot(slice_time*1e15, power_from_Eloss/1e9, label='$\Delta$E')
    sp_power.plot(slice_time*1e15, power_from_Espread/1e9, label='$\Delta\sigma_E$')

    sp_current.legend()
    sp_power.legend()

    median_image_off.plot_img_and_proj(sp_off)
    median_image_on.plot_img_and_proj(sp_on)

    lasing_dict['all_images']['Lasing_off']['image_tE'].plot_img_and_proj(sp_off_tE)
    lasing_dict['all_images']['Lasing_on']['image_tE'].plot_img_and_proj(sp_on_tE)

    lasing_dict['all_images']['Lasing_off']['image_cut'].plot_img_and_proj(sp_off_cut)
    lasing_dict['all_images']['Lasing_on']['image_cut'].plot_img_and_proj(sp_on_cut)

    sp_wake.plot(wake_t*1e15, wake_x*1e3)
    wake_profile.plot_standard(sp_profile)

    if plot_handles is None:
        plt_show()

    output = {
            'input': input_dict,
            'lasing_dict': lasing_dict,
            }
    #import pdb; pdb.set_trace()
    return output

