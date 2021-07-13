"""
No SwissFEL / PSI specific imports in this file.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt

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

rho_label = config.rho_label

def current_profile_rec_gauss(tracker, kwargs, plot_handles=None, blmeas_file=None, do_plot=True, figsize=None, both_zero_crossings=True):
    gauss_dict = tracker.find_best_gauss2(**kwargs)
    if not do_plot:
        return gauss_dict
    plot_rec_gauss(tracker, kwargs, gauss_dict, plot_handles, blmeas_file, do_plot, figsize, both_zero_crossings)
    return gauss_dict

def plot_rec_gauss(tracker, kwargs, gauss_dict, plot_handles=None, blmeas_file=None, do_plot=True, figsize=None, both_zero_crossings=True):

    #best_profile = gauss_dict['reconstructed_profile']
    #best_screen = gauss_dict['reconstructed_screen']
    best_index = gauss_dict['best_index']
    opt_func_values = gauss_dict['opt_func_values']
    opt_func_screens = gauss_dict['opt_func_screens']
    opt_func_profiles = gauss_dict['opt_func_profiles']
    opt_func_sigmas = np.array(gauss_dict['opt_func_sigmas'])
    meas_screen = gauss_dict['meas_screen']
    gauss_sigma = gauss_dict['gauss_sigma']

    if plot_handles is None:
        fig, (sp_screen, sp_profile, sp_opt, sp_moments) = reconstruction_figure(figsize)
        plt.suptitle('Optimization')
    else:
        sp_screen, sp_profile, sp_opt, sp_moments = plot_handles

    meas_screen.plot_standard(sp_screen, color='black', lw=3)

    rms_arr = np.zeros(len(opt_func_screens))
    centroid_arr = rms_arr.copy()

    for opt_ctr, (screen, profile, value, sigma) in enumerate(zip(opt_func_screens, opt_func_profiles, opt_func_values[:,1], opt_func_sigmas)):
        if opt_ctr == best_index:
            lw = 3
        else:
            lw = None
        screen.plot_standard(sp_screen, label='%.1f' % (sigma*1e15), lw=lw)
        profile.plot_standard(sp_profile, label='%.1f' % (profile.rms()*1e15), center='Mean', lw=lw)
        rms_arr[opt_ctr] = screen.rms()
        centroid_arr[opt_ctr] = screen.mean()

    #best_screen.plot_standard(sp_screen, color='red', lw=3, label='Final')
    #best_profile.plot_standard(sp_profile, color='red', lw=3, label='Final', center='Mean')

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
            blmeas_profile.plot_standard(sp_profile, ls=ls, color='black', label='%.1f' % (blmeas_profile.rms()*1e15))
            if not both_zero_crossings:
                break

    color = sp_moments.plot(opt_func_sigmas*1e15, np.abs(centroid_arr)*1e3, marker='.', label='Reconstructed centroid')[0].get_color()
    sp_moments.axhline(np.abs(meas_screen.mean())*1e3, label='Measured centroid', color=color, ls='--')
    color = sp_moments.plot(opt_func_sigmas*1e15, rms_arr*1e3, marker='.', label='Reconstructed rms')[0].get_color()
    sp_moments.axhline(meas_screen.rms()*1e3, label='Measured rms', color=color, ls='--')

    sp_moments.legend()
    sp_screen.legend(title='Initial $\sigma$ (fs)', fontsize=config.fontsize)
    sp_profile.legend(title='rms (fs)', fontsize=config.fontsize)

    yy_opt = opt_func_values[:,1]
    sp_opt.scatter(opt_func_sigmas*1e15, yy_opt)
    sp_opt.set_ylim(0,1.1*yy_opt.max())

    for sp_ in sp_opt, sp_moments:
        sp_.axvline(gauss_sigma*1e15, color='black')

    return gauss_dict

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
        #print(screen_data['x_axis'].shape)
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

    sp_proj = subplot(sp_ctr, xlabel='x (mm)', ylabel=rho_label, sciy=True)
    sp_ctr += 1
    clear_screen_calibration(sp_proj)
    return fig, (sp_proj, )

def clear_screen_calibration(sp_proj):
    for sp, title, xlabel, ylabel in [
            (sp_proj, 'Unstreaked beam', 'x (mm)', rho_label),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def reconstruction_figure(figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Current reconstruction')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,2)
    subplots = [subplot(sp_ctr) for sp_ctr in range(1, 1+4)]
    clear_reconstruction(*subplots)
    return fig, subplots

def clear_reconstruction(sp_screen, sp_profile, sp_opt, sp_moments):
    for sp, title, xlabel, ylabel in [
            (sp_screen, 'Screen', 'x (mm)', r'\rho '),
            (sp_profile, 'Profile', 't (fs)', 'Current (kA)'),
            (sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
            (sp_moments, 'Moments', 'Gaussian $\sigma$ (fs)', r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)

def lasing_figures():
    output = []
    fig = plt.figure()
    fig.canvas.set_window_title('Lasing reconstruction')
    subplot = ms.subplot_factory(3,3, grid=False)
    plot_handles = tuple((subplot(sp_ctr) for sp_ctr in range(1, 1+8)))
    output.append((fig, plot_handles))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,2, grid=False)
    plot_handles = tuple((subplot(sp_ctr) for sp_ctr in range(1, 1+4)))
    output.append((fig, plot_handles))
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
            (sp_current, 'Current', 't (fs)', 'I (kA)'),
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

def reconstruct_current(data_file_or_dict, n_streaker, beamline, tracker_kwargs_or_tracker, rec_mode, kwargs_recon, screen_x0, streaker_centers, blmeas_file=None, plot_handles=None, do_plot=True):

    if type(tracker_kwargs_or_tracker) is dict:
        tracker = tracking.Tracker(**tracker_kwargs_or_tracker)
    elif type(tracker_kwargs_or_tracker) is tracking.Tracker:
        tracker = tracker_kwargs_or_tracker
    else:
        raise ValueError(type(tracker_kwargs_or_tracker))

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
        proj_list = [median_projx]
    elif rec_mode == 'All':
        proj_list = projx

    tracker.set_simulator(meta_data)

    if x_axis[1] < x_axis[0]:
        revert = True
        x_axis = x_axis[::-1]
    else:
        revert = False

    output_dicts = []
    for proj in proj_list:
        if revert:
            proj = proj[::-1]

        meas_screen = tracking.ScreenDistribution(x_axis, proj)
        kwargs_recon['meas_screen'] = meas_screen

        #print('Analysing reconstruction')
        kwargs = copy.deepcopy(kwargs_recon)

        gaps, streaker_offsets = get_gap_and_offset(meta_data, beamline)

        kwargs['meas_screen']._xx = kwargs['meas_screen']._xx - screen_x0
        kwargs['beam_offsets'] = -(streaker_offsets - streaker_centers)
        kwargs['gaps'] = gaps
        kwargs['meas_screen'].cutoff2(tracker.screen_cutoff)
        kwargs['meas_screen'].crop()
        kwargs['meas_screen'].reshape(tracker.len_screen)
        kwargs['n_streaker'] = n_streaker

        # Only allow one streaker at the moment
        for n in (0,1):
            if n != kwargs['n_streaker']:
                kwargs['beam_offsets'][n] = 0

        gauss_dict = current_profile_rec_gauss(tracker, kwargs, plot_handles, blmeas_file, do_plot=do_plot)
        output_dict = {
                'input': {
                    'data_file_or_dict': data_file_or_dict,
                    'n_streaker': n_streaker,
                    'beamline': beamline,
                    'tracker_kwargs': tracker_kwargs_or_tracker,
                    'rec_mode': rec_mode,
                    'kwargs_recon': kwargs_recon,
                    'screen_x0': screen_x0,
                    'streaker_centers': streaker_centers,
                    'blmeas_file': blmeas_file,
                    },
                'gauss_dict': gauss_dict,
                }
        output_dicts.append(output_dict)

    if rec_mode == 'Median':
        return output_dict
    elif rec_mode == 'All':
        return output_dicts
    else:
        print(rec_mode)

def get_gap_and_offset(meta_data, beamline):
    gaps, offsets = [], []
    for streaker in config.streaker_names[beamline].values():
        gaps.append(meta_data[streaker+':GAP']*1e-3)
        offsets.append(meta_data[streaker+':CENTER']*1e-3)

    return np.array(gaps), np.array(offsets)

def get_beamline_n_streaker(streaker):
    for beamline, d in config.streaker_names.items():
        for n_streaker, streaker2 in d.items():
            if streaker == streaker2:
                return beamline, n_streaker

