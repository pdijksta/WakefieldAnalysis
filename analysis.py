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


def current_profile_rec_gauss(tracker, kwargs, plot_handles=None, blmeas_file=None):

    gauss_dict = tracker.find_best_gauss(**kwargs)
    #import pickle
    #with open('/tmp/tmp_gauss_dict.pkl', 'wb') as f:
    #    pickle.dump((gauss_dict, self.tracker_args, kwargs), f)

    print('Do plotting')
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
                blmeas_profile = iap.profile_from_blmeas(blmeas_file, kwargs['tt_halfrange'], kwargs['charge'], tracker.energy_eV, True, zero_crossing)
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
        plt.show()

    return gauss_dict

def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, const, semigap):
    wall0, wall1 = -semigap, semigap
    c1 = np.abs((offsets-streaker_offset-wall0))**(-order)
    c2 = np.abs((offsets-streaker_offset-wall1))**(-order)
    return const + (c1 - c2)*strength

def analyze_streaker_calibration(filename_or_dict, do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None, forward_propagate_blmeas=False, tracker=None, blmeas=None, beamline='Aramis', charge=200e-12, fit_gap=False, debug=False):
    if type(filename_or_dict) is dict:
        data_dict = filename_or_dict
    elif type(filename_or_dict) is str:
        data_dict = h5_storage.loadH5Recursive(filename_or_dict)
    else:
        raise ValueError(type(filename_or_dict))

    result_dict = data_dict['pyscan_result']

    if 'image' in result_dict:
        try:
            images = result_dict['image'].astype(float).squeeze()
        except:
            import pdb; pdb.set_trace()
        proj_x = np.sum(images, axis=-2).squeeze()
    elif 'projx' in result_dict:
        proj_x = result_dict['projx']

    if 'x_axis_m' in result_dict:
        x_axis = result_dict['x_axis_m']
    elif 'x_axis' in result_dict:
        x_axis = result_dict['x_axis']
    else:
        print(result_dict.keys())
        raise KeyError

    offsets = data_dict['streaker_offsets'].squeeze()
    n_images = int(data_dict['n_images'])

    centroids = np.zeros([len(offsets), n_images])
    rms = np.zeros_like(centroids)

    plot_list = []
    if n_images > 1:
        for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
            proj = proj_x[n_o,n_i]
            proj -= np.median(proj)
            proj[np.abs(proj)< np.abs(proj).max()*0.02] = 0
            if n_i == 0:
                plot_list.append(proj)
            centroids[n_o,n_i] = cc = np.sum(proj*x_axis) / np.sum(proj)
            rms[n_o, n_i] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
        centroid_mean = np.mean(centroids, axis=1)
        centroid_std = np.std(centroids, axis=1)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)
    elif n_images == 1:
        for n_o in range(len(offsets)):
            proj = proj_x[n_o]
            proj -= np.median(proj)
            proj[np.abs(proj)< np.abs(proj).max()*0.05] = 0
            centroids[n_o] = cc = np.sum(proj*x_axis) / np.sum(proj)
            rms[n_o] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
            plot_list.append(proj)
        centroid_mean = centroids.squeeze()
        centroid_std = None
        rms_mean = rms
        rms_std = None

    streaker = data_dict['streaker']
    if 'meta_data_end' in data_dict:
        key = 'meta_data_end'
    else:
        key = 'meta_data'
    meta_data0 = data_dict[key]
    semigap = meta_data0[streaker+':GAP']/2.*1e-3


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
    if fit_gap:
        p0.append(semigap)

    def fit_func(*args):
        if fit_order and not fit_gap:
            return streaker_calibration_fit_func(*args, const0, semigap)
        if not fit_order and not fit_gap:
            return streaker_calibration_fit_func(*args, order0, const0, semigap)
        if fit_order and fit_gap:
            new_args = args[:-1] + (const0, args[-1])
            return streaker_calibration_fit_func(*new_args)
        if not fit_order and fit_gap:
            new_args = args[:-1] + (order0, const0, args[-1])
            return streaker_calibration_fit_func(*new_args)
    try:
        p_opt, p_cov = curve_fit(fit_func, offsets, centroid_mean, p0, sigma=centroid_std)
    except RuntimeError:
        print('Streaker calibration did not converge')
        p_opt = p0
        if debug:
            fignum = plt.gcf().number
            plt.figure()
            plt.plot(offsets, centroid_mean, ls='None', marker='.')
            fit_xx = np.linspace(offsets.min(), offsets.max(), 100)
            plt.plot(fit_xx, fit_func(fit_xx, *p0), ls='--')
            plt.show()
            plt.figure(fignum)

    xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
    reconstruction = fit_func(xx_fit, *p_opt)
    initial_guess = fit_func(xx_fit, *p0)
    streaker_offset = p_opt[0]
    if fit_gap:
        gap_fit = p_opt[-1]*2
    else:
        gap_fit = semigap*2

    if fit_order:
        order_fit = p_opt[2]
    else:
        order_fit = order0

    meas_screens = []
    for n_proj, (proj, offset) in enumerate(zip(plot_list, offsets)):
        meas_screen = misc.proj_to_screen(proj, x_axis, False, const0)
        len_screen = len(meas_screen.x)
        meas_screen.cutoff2(3e-2)
        meas_screen.crop()
        meas_screen.reshape(len_screen)
        meas_screens.append(meas_screen)

    meta_data = {
            'p_opt': p_opt,
            'p0': p0,
            'centroids': centroids,
            'centroid_mean': centroid_mean,
            'centroid_std': centroid_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'offsets': offsets,
            'semigap': semigap,
            'streaker_offset': streaker_offset,
            'fit_reconstruction': reconstruction,
            'fit_xx': xx_fit,
            'screen_x0': const0,
            'gap_fit': gap_fit,
            'order_fit': order_fit,
            }


    if forward_propagate_blmeas:
        blmeas_profile = iap.profile_from_blmeas(blmeas, 200e-15, charge, tracker.energy_eV)
        blmeas_profile.cutoff2(5e-2)
        blmeas_profile.crop()
        blmeas_profile.reshape(len_screen)

        streaker_names = config.streaker_names[beamline]
        gaps = [meta_data0[x+':GAP']*1e-3 for x in streaker_names.values()]
        streaker_offsets0 = []
        n_streaker_var = None
        for n_streaker, streaker in streaker_names.items():
            if streaker == data_dict['streaker']:
                n_streaker_var = n_streaker
                streaker_offsets0.append(None)
            else:
                streaker_offsets0.append(meta_data0[streaker+':CENTER']*1e-3)
        assert n_streaker_var is not None

        sim_screens = []
        for s_offset in offsets:
            streaker_offsets = streaker_offsets0[:]
            streaker_offsets[n_streaker_var] = -(s_offset-streaker_offset)
            forward_dict = tracker.matrix_forward(blmeas_profile, gaps, streaker_offsets)
            sim_screen = forward_dict['screen']
            sim_screen.cutoff2(3e-2)
            sim_screen.crop()
            sim_screen.reshape(len_screen)
            sim_screens.append(sim_screen)
        meta_data['sim_screens'] = sim_screens
    else:
        sim_screens = None

    if do_plot:

        if plot_handles is None:
            fig, (sp_center, sp_sizes, sp_proj, sp_current) = streaker_calibration_figure()
        else:
            (sp_center, sp_sizes, sp_proj, sp_current) = plot_handles
        screen = data_dict['screen']
        sp_center.set_title(screen)

        xx_plot = (offsets - streaker_offset)*1e3
        xx_plot_fit = (xx_fit - streaker_offset)*1e3
        sp_center.errorbar(xx_plot, (centroid_mean-const0)*1e3, yerr=centroid_std*1e3, label='Data', ls='None', marker='o')
        sp_sizes.errorbar(xx_plot, (rms_mean-const0)*1e3, yerr=rms_std*1e3, label='Data', marker='o')

        sp_center.plot(xx_plot_fit, (reconstruction-const0)*1e3, label='Fit')
        sp_center.plot(xx_plot_fit, (initial_guess-const0)*1e3, label='Guess')
        sp_center.legend()

        for n_proj, (meas_screen, offset) in enumerate(zip(meas_screens, offsets)):
            color = ms.colorprog(n_proj, offsets)
            meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
            if sim_screens is not None:
                sim_screen = sim_screens[n_proj]
                sim_screen.plot_standard(sp_proj, color=color, ls='--')

        sp_proj.legend()

        if forward_propagate_blmeas:
            blmeas_profile.plot_standard(sp_current, color='black')

    output = {
            'raw_data': data_dict,
            'meta_data': meta_data,
            }

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
        x_axis = screen_data['x_axis'].squeeze()*1e-6
        if len(x_axis.shape) == 2:
            x_axis = x_axis[0]

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
    subplot = ms.subplot_factory(2, 2)
    sp_center = subplot(sp_ctr)
    sp_ctr +=1
    sp_sizes = subplot(sp_ctr)
    sp_ctr += 1
    sp_proj = subplot(sp_ctr)
    sp_ctr += 1
    sp_current = subplot(sp_ctr)
    clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_current)
    return fig, (sp_center, sp_sizes, sp_proj, sp_current)

def clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_current):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Centroid shift', 'Streaker center [mm]', 'Beam X centroid [mm]'),
            (sp_sizes, 'Size increase', 'Streaker center [mm]', 'Beam X rms [mm]'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            (sp_current, 'Beam current', 't (fs)', 'Current (kA)'),
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
    sp_ctr += 1
    sp_on = subplot(sp_ctr)
    sp_ctr += 1
    sp_on_cut = subplot(sp_ctr)
    sp_ctr += 1
    sp_on_tE = subplot(sp_ctr)
    sp_ctr += 1
    sp_off = subplot(sp_ctr)
    sp_ctr += 1
    sp_off_cut = subplot(sp_ctr)
    sp_ctr += 1
    sp_off_tE = subplot(sp_ctr)
    sp_ctr += 1
    output.append((fig, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,2, grid=False)
    sp_ctr = 1
    sp_power = subplot(sp_ctr)
    sp_ctr += 1
    sp_current = subplot(sp_ctr)
    sp_ctr += 1
    sp_centroid = subplot(sp_ctr)
    sp_ctr += 1
    sp_slice_size = subplot(sp_ctr)
    sp_ctr += 1
    output.append((fig, (sp_power, sp_current, sp_centroid, sp_slice_size)))
    clear_lasing(output)
    return output

def clear_lasing(plot_handles):
    (_, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)) = plot_handles[0]
    (_, (sp_power, sp_current, sp_centroid, sp_slice_size)) = plot_handles[1]

    for sp, title, xlabel, ylabel in [
            (sp_profile, 'Current profile', 't (fs)', 'I (kA)'),
            (sp_wake, 'Wake', 't (fs)', 'x (mm)'),
            (sp_off, 'Lasing off raw', 'x (mm)', 'y (mm)'),
            (sp_on, 'Lasing on raw', 'x (mm)', 'y (mm)'),
            (sp_off_cut, 'Lasing off cut', 'x (mm)', 'y (mm)'),
            (sp_on_cut, 'Lasing on cut', 'x (mm)', 'y (mm)'),
            (sp_off_tE, 'Lasing off tE', 't (fs)', '$\Delta$ E (MeV)'),
            (sp_on_tE, 'Lasing on tE', 't (fs)', '$\Delta$ E (MeV)'),
            (sp_power, 'Power', 't (fs)', 'P (GW)'),
            (sp_current, 'Current', 't (fs)', 'I (arb. units)'),
            (sp_centroid, 'Slice centroids', 't (fs)', '$\Delta$ E (MeV)'),
            (sp_slice_size, 'Slice sizes', 't (fs)', '$\sigma_E$ (MeV)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)

def reconstruct_lasing(file_or_dict_on, file_or_dict_off, screen_center, structure_center, structure_length, file_current, r12, disp, energy_eV, charge, streaker, plot_handles, pulse_energy, n_slices, len_profile):

    if type(file_or_dict_on) is dict:
        dict_on = file_or_dict_on
    else:
        dict_on = h5_storage.loadH5Recursive(file_or_dict_on)
    dict_on_p = dict_on['pyscan_result']
    dict_on_m = dict_on['meta_data_end']

    if type(file_or_dict_off) is dict:
        dict_off = file_or_dict_off
    else:
        dict_off = h5_storage.loadH5Recursive(file_or_dict_off)
    dict_off_p = dict_off['pyscan_result']
    dict_off_m = dict_off['meta_data_end']

    if energy_eV == 'file':
        if 'SARBD01-MBND100:ENERGY-OP' in dict_on_m:
            energy_eV = dict_on_m['SARBD01-MBND100:ENERGY-OP']*1e6
        elif 'SARBD01-MBND100:P-SET' in dict_on_m:
            energy_eV = dict_on_m['SARBD01-MBND100:P-SET']*1e6
        else:
            raise ValueError('No energy saved!')


    input_dict = {
            'file_or_dict_on': file_or_dict_on,
            'file_or_dict_off': file_or_dict_off,
            'screen_center': screen_center,
            'structure_center': structure_center,
            'structure_length': structure_length,
            'file_current': file_current,
            'r12': r12,
            'disp': disp,
            'energy_eV': energy_eV,
            'charge': charge,
            'streaker': streaker,
            'pulse_energy': pulse_energy,
            }

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

    current_dict = h5_storage.loadH5Recursive(file_current)
    if 'gaussian_reconstruction' in current_dict:
        wake_profile_dict = current_dict['gaussian_reconstruction']['reconstructed_profile']
        wake_profile = iap.BeamProfile.from_dict(wake_profile_dict)
    else:
        wake_profile = iap.profile_from_blmeas(current_dict, 200e-15, charge, energy_eV, True)

    wake_profile.cutoff2(0.1)
    wake_profile.crop()
    wake_profile.reshape(len_profile)

    wake_t, wake_x = wake_profile.get_x_t(gap, beam_offset, structure_length, r12)

    lasing_dict = lasing.obtain_lasing(median_image_off, median_image_on, n_slices, wake_x, wake_t, len_profile, disp, energy_eV, charge, pulse_energy=pulse_energy, debug=False)

    if plot_handles is None:
        plot_handles = lasing_figures()

    (fig, (sp_profile, sp_wake, sp_off, sp_on, sp_off_cut, sp_on_cut, sp_off_tE, sp_on_tE)) = plot_handles[0]
    (fig, (sp_power, sp_current, sp_centroid, sp_slice_size)) = plot_handles[1]

    slice_time = lasing_dict['slice_time']
    all_slice_dict = lasing_dict['all_slice_dict']
    power_from_Eloss = lasing_dict['power_Eloss']
    power_from_Espread = lasing_dict['power_Espread']

    sp_current.plot(slice_time*1e15, all_slice_dict['Lasing_off']['slice_current'], label='Off')
    sp_current.plot(slice_time*1e15, all_slice_dict['Lasing_on']['slice_current'], label='On')

    lasing_dict['all_images']['Lasing_off']['image_tE'].plot_img_and_proj(sp_off_tE)
    lasing_dict['all_images']['Lasing_on']['image_tE'].plot_img_and_proj(sp_on_tE)

    lasing_dict['all_images']['Lasing_off']['image_cut'].plot_img_and_proj(sp_off_cut)
    lasing_dict['all_images']['Lasing_on']['image_cut'].plot_img_and_proj(sp_on_cut)

    for key, sp_tE in [('Lasing_off',sp_off_tE), ('Lasing_on',sp_on_tE)]:
        slice_sigma = all_slice_dict[key]['slice_sigma']
        slice_centroid = all_slice_dict[key]['slice_mean']
        sp_slice_size.plot(slice_time*1e15, slice_sigma/1e6, label=key)
        sp_centroid.plot(slice_time*1e15, slice_centroid/1e6, label=key)

        lims = sp_tE.get_ylim()
        sp_tE.errorbar(slice_time*1e15, slice_centroid/1e6, yerr=slice_sigma/1e6, ls='None', marker='+', color='red')
        sp_tE.set_ylim(*lims)

    sp_power.plot(slice_time*1e15, power_from_Eloss/1e9, label='$\Delta$E')
    sp_power.plot(slice_time*1e15, power_from_Espread/1e9, label='$\Delta\sigma_E$')

    sp_current.legend()
    sp_power.legend()
    sp_slice_size.legend()
    sp_centroid.legend()

    median_image_off.plot_img_and_proj(sp_off)
    median_image_on.plot_img_and_proj(sp_on)

    sp_wake.plot(wake_t*1e15, wake_x*1e3)
    wake_profile.plot_standard(sp_profile)

    if plot_handles is None:
        plt.show()

    if type(input_dict['file_or_dict_on']) is dict:
        del input_dict['file_or_dict_on']
    if type(input_dict['file_or_dict_off']) is dict:
        del input_dict['file_or_dict_off']

    output = {
            'input': input_dict,
            'lasing_dict': lasing_dict,
            }
    #import pdb; pdb.set_trace()
    return output

def reconstruct_current(data_file_or_dict, n_streaker, beamline, tracker_kwargs, rec_mode, kwargs_recon, screen_x0, streaker_centers, blmeas_file=None, plot_handles=None):

    tracker = tracking.Tracker(**tracker_kwargs)

    if type(data_file_or_dict) is dict:
        screen_data = data_file_or_dict
    else:
        screen_data = h5_storage.loadH5Recursive(data_file_or_dict)

    if 'meta_data' in screen_data:
        meta_data = screen_data['meta_data']
    elif 'meta_data_begin' in screen_data:
        meta_data = screen_data['meta_data_begin']
    else:
        print(screen_data.keys())
        raise ValueError

    if 'pyscan_result' in screen_data:
        pyscan_data = screen_data['pyscan_result']
    else:
        pyscan_data = screen_data

    x_axis = pyscan_data['x_axis_m']
    projx = pyscan_data['image'].sum(axis=-2)
    if rec_mode == 'Median':
        median_projx = misc.get_median(projx)
    elif rec_mode == 'All':
        raise NotImplementedError

    tracker.set_simulator(meta_data)

    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        median_projx = median_projx[::-1]

    meas_screen = tracking.ScreenDistribution(x_axis, median_projx)
    kwargs_recon['meas_screen'] = meas_screen

    print('Analysing reconstruction')

    kwargs = copy.deepcopy(kwargs_recon)

    gaps, streaker_offsets = get_gap_and_offset(meta_data, beamline)

    kwargs['meas_screen']._xx = kwargs['meas_screen']._xx - screen_x0
    kwargs['beam_offsets'] = -(streaker_offsets - streaker_centers)
    kwargs['gaps'] = gaps
    kwargs['meas_screen'].cutoff2(tracker.screen_cutoff)
    kwargs['meas_screen'].crop()
    kwargs['meas_screen'].reshape(tracker.len_screen)

    # Only allow one streaker at the moment
    for n in (0,1):
        if n != kwargs['n_streaker']:
            kwargs['beam_offsets'][n] = 0

    gauss_dict = current_profile_rec_gauss(tracker, kwargs, plot_handles, blmeas_file)

    output_dict = {
            'input': {
                'data_file_or_dict': data_file_or_dict,
                'n_streaker': n_streaker,
                'beamline': beamline,
                'tracker_kwargs': tracker_kwargs,
                'rec_mode': rec_mode,
                'kwargs_recon': kwargs_recon,
                'screen_x0': screen_x0,
                'streaker_centers': streaker_centers,
                'blmeas_file': blmeas_file,
                },
            'gauss_dict': gauss_dict,
            }

    return output_dict

def get_gap_and_offset(meta_data, beamline):
    gaps, offsets = [], []
    for streaker in config.streaker_names[beamline].values():
        gaps.append(meta_data[streaker+':GAP']*1e-3)
        offsets.append(meta_data[streaker+':CENTER']*1e-3)

    return np.array(gaps), np.array(offsets)

