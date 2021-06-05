import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import h5_storage
    import myplotstyle as ms
    import misc2 as misc
    import config
    import image_and_profile as iap
except ImportError:
    from . import h5_storage
    from . import myplotstyle as ms
    from . import misc2 as misc
    from . import config
    from . import image_and_profile as iap

order0_centroid = 2.75
order0_rms = 2.75

def analyze_streaker_calibration(result_dict, do_plot=True, plot_handles=None, fit_order=False, forward_propagate_blmeas=False, tracker=None, blmeas=None, beamline='Aramis', charge=None, fit_gap=True, tt_halfrange=None, force_gap=None):
    meta_data = result_dict['meta_data_begin']
    streakers = list(config.streaker_names[beamline].values())
    offsets = np.array([meta_data[x+':CENTER'] for x in streakers])
    n_streaker = int(np.argmax(np.abs(offsets)).squeeze())

    if force_gap is None:
        gap0 = meta_data[streakers[n_streaker]+':GAP']*1e-3
    else:
        gap0 = force_gap
    if charge is None:
        charge = meta_data[config.beamline_chargepv[beamline]]*1e-12
    if tt_halfrange is None:
        tt_halfrange = config.get_default_gauss_recon_settings()['tt_halfrange']

    sc = StreakerCalibration(beamline, n_streaker, gap0, file_or_dict=result_dict)
    sc.fit()
    if forward_propagate_blmeas:
        beam_profile = iap.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, True, 1)
        beam_profile.reshape(tracker.len_screen)
        beam_profile.cutoff2(5e-2)
        beam_profile.crop()
        beam_profile.reshape(tracker.len_screen)

        sc.forward_propagate(beam_profile, tt_halfrange, charge, tracker)
    sc.plot_streaker_calib(plot_handles)
    return sc.get_result_dict()

class StreakerCalibration:

    def __init__(self, beamline, n_streaker, gap0, file_or_dict=None, offsets_range=None, images=None, x_axis=None, fit_gap=True, fit_order=False, order_centroid=order0_centroid, order_rms=order0_rms, proj_cutoff=0.03):
        self.order_rms = order_rms
        self.order_centroid = order_centroid
        self.fit_gap = fit_gap
        self.fit_order = fit_order
        self.gap0 = gap0
        self.n_streaker = n_streaker
        self.proj_cutoff = proj_cutoff
        self.beamline = beamline

        self.offsets = []
        self.screen_x0_arr = []
        self.centroids = []
        self.centroids_std = []
        self.rms = []
        self.rms_std = []
        self.images = None
        self.blmeas_profile = None
        self.sim_screen_dict = {}
        self.sim_screens = None
        self.plot_list_x = []
        self.plot_list_y = []
        self.raw_data = None

        self.fit_dicts_gap_order = {
                'beamsize':{
                    True: {
                        True: None,
                        False: None,
                    },
                    False: {
                        True: None,
                        False: None,
                    },
                },
                'centroid':{
                    True: {
                        True: None,
                        False: None,
                    },
                    False: {
                        True: None,
                        False: None,
                    },
                },
                }
        self.gauss_dicts_gap_order = copy.deepcopy(self.fit_dicts_gap_order)

        if offsets_range is not None:
            self.add_data(offsets_range, images, x_axis)
        if file_or_dict is not None:
            self.add_file(file_or_dict)

    def get_result_dict(self):
        fit_dict_centroid = self.fit_dicts_gap_order['centroid'][self.fit_gap][self.fit_order]
        fit_dict_rms = self.fit_dicts_gap_order['beamsize'][self.fit_gap][self.fit_order]
        meta_data = {
                 'centroid_mean': self.centroids,
                 'centroid_std': self.centroids_std,
                 'rms_mean': self.rms,
                 'rms_std': self.rms_std,
                 'offsets': self.offsets,
                 'semigap': fit_dict_centroid['gap_fit']/2.,
                 'streaker_offset': fit_dict_centroid['streaker_offset'],
                 'x_axis': self.plot_list_x[0],
                 'streaker': config.streaker_names[self.beamline][self.n_streaker],
                 'fit_dict_rms': fit_dict_rms,
                 'fit_dict_centroid': fit_dict_centroid
                 }
        output = {
                'raw_data': self.raw_data,
                'meta_data': meta_data,
                }
        return output

    def add_data(self, offsets, images, x_axis):
        n_images = images.shape[1]

        centroids = np.zeros([len(offsets), n_images])
        rms = np.zeros_like(centroids)
        proj_x = images.astype(np.float64).sum(axis=-2)

        where0 = np.argwhere(offsets == 0).squeeze()
        assert where0.size == 1

        plot_list_y = []
        for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
            proj = proj_x[n_o,n_i]
            proj = proj - np.median(proj)
            proj[proj<proj.max()*self.proj_cutoff] = 0
            if n_i == 0:
                plot_list_y.append(proj)
            centroids[n_o,n_i] = cc = np.sum(proj*x_axis) / np.sum(proj)
            rms[n_o, n_i] = np.sqrt(np.sum(proj*(x_axis-cc)**2) / np.sum(proj))
            if np.isnan(rms[n_o, n_i]):
                import pdb; pdb.set_trace()
        centroid_mean = np.mean(centroids, axis=1)
        screen_x0 = centroid_mean[where0]
        centroid_mean -= screen_x0
        screen_x0_arr = np.array([screen_x0]*len(offsets), float)
        centroid_std = np.std(centroids, axis=1)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

        if 0 in self.offsets:
            mask = offsets != 0
        else:
            mask = np.ones_like(offsets, dtype=bool)

        new_offsets = np.concatenate([self.offsets, offsets[mask]])
        sort = np.argsort(new_offsets)
        self.offsets = new_offsets[sort]
        self.centroids = np.concatenate([self.centroids, centroid_mean[mask]])[sort]
        self.centroids_std = np.concatenate([self.centroids_std, centroid_std[mask]])[sort]
        self.rms = np.concatenate([self.rms, rms_mean[mask]])[sort]
        self.rms_std = np.concatenate([self.rms_std, rms_std[mask]])[sort]
        self.screen_x0_arr = np.concatenate([self.screen_x0_arr, screen_x0_arr[mask]])[sort]

        plot_list_x = [x_axis - screen_x0] * len(plot_list_y)
        new_plot_list_x = self.plot_list_x + plot_list_x
        new_plot_list_y = self.plot_list_y + plot_list_y

        self.plot_list_x = new_plot_list_x[:]
        self.plot_list_y = new_plot_list_y[:]
        for index, new_index in enumerate(sort):
            self.plot_list_x[index] = new_plot_list_x[new_index]
            self.plot_list_y[index] = new_plot_list_y[new_index]

        if self.images is None:
            self.images = images[mask]
        else:
            self.images = np.concatenate([self.images, images[mask]])[sort]

    def add_file(self, filename_or_dict):
        if type(filename_or_dict) is dict:
            data_dict = filename_or_dict
        elif type(filename_or_dict) is str:
            data_dict = h5_storage.loadH5Recursive(filename_or_dict)
        else:
            raise ValueError(type(filename_or_dict))

        if 'raw_data' in data_dict:
            data_dict = data_dict['raw_data']
        if 'meta_data_begin' in data_dict:
            self.meta_data = data_dict['meta_data_begin']
        result_dict = data_dict['pyscan_result']
        images = result_dict['image'].squeeze()
        if 'x_axis_m' in result_dict:
            x_axis = result_dict['x_axis_m']
        elif 'x_axis' in result_dict:
            x_axis = result_dict['x_axis']
        else:
            print(result_dict.keys())
            raise KeyError

        offsets = data_dict['streaker_offsets'].squeeze()
        self.add_data(offsets, images, x_axis)
        self.raw_data = data_dict

    @staticmethod
    def beamsize_fit_func(offsets, streaker_offset, strength, order, semigap, const):
        sq0 = const**2
        c1 = np.abs((offsets-streaker_offset+semigap))**(-order*2)
        c2 = np.abs((offsets-streaker_offset-semigap))**(-order*2)
        if strength > 0:
            sq_add = strength**2 * (c1+c2)
        else:
            sq_add = np.zeros(len(offsets))
        output = np.sqrt(sq0 + sq_add)
        return output

    @staticmethod
    def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, semigap, const):
        c1 = np.abs((offsets-streaker_offset+semigap))**(-order)
        c2 = np.abs((offsets-streaker_offset-semigap))**(-order)
        return const + (c1 - c2)*strength

    def fit_type(self, type_):

        offsets = self.offsets
        semigap = self.gap0/2.
        where0 = np.argwhere(offsets == 0).squeeze()

        if type_ == 'beamsize':
            yy_mean = self.rms
            yy_std = self.rms_std
            fit_func = self.beamsize_fit_func
            order0 = self.order_rms
        elif type_ == 'centroid':
            yy_mean = self.centroids
            yy_std = self.centroids_std
            fit_func = self.streaker_calibration_fit_func
            order0 = self.order_centroid

        const0 = yy_mean[where0]
        offset0 = (offsets[0] + offsets[-1])/2

        s0_arr = yy_mean/(np.abs((offsets-offset0-semigap))**(-order0) + np.abs((offsets-offset0+semigap))**(-order0))
        s0 = (s0_arr[0] + s0_arr[-1])/2
        p0 = [offset0, s0]
        if self.fit_order:
            p0.append(order0)
        if self.fit_gap:
            p0.append(semigap)

        def fit_func2(*args):
            args = list(args)
            if self.fit_order:
                if self.fit_gap:
                    output = fit_func(*args, const0)
                else:
                    output = fit_func(*args, semigap, const0)
            else:
                if self.fit_gap:
                    output = fit_func(*args[:-1], order0, args[-1], const0)
                else:
                    output = fit_func(*args, order0, semigap, const0)
            return output

        try:
            p_opt, p_cov = curve_fit(fit_func2, offsets, yy_mean, p0, sigma=yy_std)
        except RuntimeError:
            print('Streaker calibration type %s did not converge' % type_)
            p_opt = p0

        streaker_offset = p_opt[0]
        if self.fit_gap:
            gap_fit = p_opt[-1]*2
        else:
            gap_fit = abs(semigap*2)
        if self.fit_order:
            order_fit = p_opt[2]
        else:
            order_fit = order0

        xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
        xx_fit2_min = -(gap_fit/2-streaker_offset-10e-6)
        xx_fit2_max = -xx_fit2_min + 2*streaker_offset
        xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
        reconstruction = fit_func2(xx_fit, *p_opt)
        reconstruction2 = fit_func2(xx_fit2, *p_opt)
        initial_guess = fit_func2(xx_fit, *p0)
        fit_dict = {
                'reconstruction': reconstruction,
                'reconstruction2': reconstruction2,
                'initial_guess': initial_guess,
                'streaker_offset': streaker_offset,
                'gap_fit': gap_fit,
                'order_fit': order_fit,
                'p_opt': p_opt,
                'p0': p0,
                'xx_fit': xx_fit,
                'xx_fit2': xx_fit2,
                'screen_rms0': const0
                }
        self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order] = fit_dict
        return fit_dict

    def fit(self):
        self.fit_type('beamsize')
        self.fit_type('centroid')

    def forward_propagate(self, blmeas_profile, tt_halfrange, charge, tracker, type_='centroid', blmeas_cutoff=None):
        tracker.set_simulator(self.meta_data)
        streaker_offset = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['streaker_offset']
        gap = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['gap_fit']
        offsets = self.offsets
        if type(blmeas_profile) is iap.BeamProfile:
            pass
        else:
            blmeas_profile = iap.profile_from_blmeas(blmeas_profile, tt_halfrange, charge, tracker.energy_eV, True)
            if blmeas_cutoff is None:
                blmeas_profile.cutoff2(tracker.profile_cutoff)
            else:
                blmeas_profile.cutoff2(blmeas_cutoff)
            blmeas_profile.crop()
            blmeas_profile.reshape(tracker.len_screen)

        len_screen = tracker.len_screen
        gaps = np.array([10., 10.])
        gaps[self.n_streaker] = gap
        beam_offsets0 = np.array([0., 0.])

        sim_screens = []
        for s_offset in offsets:
            beam_offsets = beam_offsets0[:]
            beam_offsets[self.n_streaker] = -(s_offset-streaker_offset)
            forward_dict = tracker.matrix_forward(blmeas_profile, gaps, beam_offsets)
            sim_screen = forward_dict['screen']
            sim_screen.cutoff2(tracker.screen_cutoff)
            sim_screen.crop()
            sim_screen.reshape(len_screen)
            sim_screens.append(sim_screen)

        self.blmeas_profile = blmeas_profile
        self.sim_screen_dict[(gap, streaker_offset)] = sim_screens
        self.sim_screens = sim_screens
        return blmeas_profile, sim_screens

    def plot_streaker_calib(self, plot_handles=None):

        if plot_handles is None:
            fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = streaker_calibration_figure()
        else:
            (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = plot_handles

        offsets = self.offsets
        fit_dict_centroid = self.fit_dicts_gap_order['centroid'][self.fit_gap][self.fit_order]
        fit_dict_rms = self.fit_dicts_gap_order['beamsize'][self.fit_gap][self.fit_order]
        blmeas_profile = self.blmeas_profile
        forward_propagate_blmeas = (blmeas_profile is not None)
        screen_x0 = 0

        meas_screens = []
        for n_proj, (x_axis, proj, offset) in enumerate(zip(self.plot_list_x, self.plot_list_y, offsets)):
            meas_screen = misc.proj_to_screen(proj, x_axis, False, screen_x0)
            meas_screen.cutoff2(3e-2)
            meas_screen.crop()
            meas_screen.reshape(int(1e3))
            meas_screens.append(meas_screen)

        rms_sim = np.zeros(len(offsets))
        centroid_sim = np.zeros(len(offsets))
        if self.sim_screens is not None:
            sim_screens = self.sim_screens
            #len_screen = len(sim_screens[0])
            for n_proj, (meas_screen, offset) in enumerate(zip(meas_screens, offsets)):
                color = ms.colorprog(n_proj, offsets)
                meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
                if sim_screens is not None:
                    sim_screen = sim_screens[n_proj]
                    sim_screen.plot_standard(sp_proj, color=color, ls='--')
                    centroid_sim[n_proj] = sim_screen.mean()
                    rms_sim[n_proj] = sim_screen.rms()
        else:
            sim_screens = None

        if forward_propagate_blmeas:
            blmeas_profile.plot_standard(sp_current, color='black')

        for fit_dict, sp1, sp2, yy, yy_err, yy_sim in [
                (fit_dict_centroid, sp_center, sp_center2, self.centroids, self.centroids_std, centroid_sim),
                (fit_dict_rms, sp_sizes, sp_sizes2, self.rms, self.rms_std, rms_sim),
                ]:

            xx_fit = fit_dict['xx_fit']
            xx_fit2 = fit_dict['xx_fit2']
            reconstruction = fit_dict['reconstruction']
            reconstruction2 = fit_dict['reconstruction2']
            gap = fit_dict['gap_fit']
            fit_semigap = gap/2
            streaker_offset = fit_dict['streaker_offset']

            xx_plot = (offsets - streaker_offset)
            xx_plot_fit = (xx_fit - streaker_offset)
            sp1.errorbar(xx_plot*1e3, (yy-screen_x0)*1e3, yerr=yy_err*1e3, label='Data', ls='None', marker='o')
            sp1.plot(xx_plot_fit*1e3, (reconstruction-screen_x0)*1e3, label='Fit')

            mask_pos, mask_neg = offsets > 0, offsets < 0
            xx_plot2 = np.abs(fit_semigap - np.abs(xx_plot))
            for mask2, label in [(mask_pos, 'Positive'), (mask_neg, 'Negative')]:
                sp2.errorbar(xx_plot2[mask2]*1e6, np.abs(yy[mask2]-screen_x0)*1e3, yerr=yy_err[mask2]*1e3, label=label, marker='o', ls='None')

            if sim_screens is not None:
                plot2_sim = []
                for mask in mask_pos, mask_neg:
                    plot2_sim.extend([(a, np.abs(b)) for a, b in zip(xx_plot2[mask], yy_sim[mask])])
                plot2_sim.sort()
                xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
                xx_plot_sim = np.array(xx_plot_sim)
                yy_plot_sim = np.array(yy_plot_sim)
                sp2.plot(xx_plot_sim*1e6, yy_plot_sim*1e3, label='Simulated', ls='None', marker='o')
                sp1.plot(xx_plot*1e3, yy_sim*1e3, label='Simulated', marker='.', ls='None')

            xx_plot_fit2 = np.abs(fit_semigap - np.abs(xx_fit2 - streaker_offset))
            yy_plot_fit2 = np.abs(reconstruction2)-screen_x0
            xlims = sp_center2.get_xlim()
            mask_fit = np.logical_and(xx_plot_fit2*1e6 > xlims[0], xx_plot_fit2*1e6 < xlims[1])
            mask_fit = np.logical_and(mask_fit, xx_fit2 > 0)
            sp2.plot(xx_plot_fit2[mask_fit]*1e6, yy_plot_fit2[mask_fit]*1e3, label='Fit')
            sp2.set_xlim(*xlims)

            title = sp1.get_title()
            sp1.set_title('%s; Gap=%.2f mm' % (title, fit_dict['gap_fit']*1e3), fontsize=config.fontsize)
            title = sp2.get_title()
            sp2.set_title('%s; Center=%i $\mu$m' % (title, round(fit_dict['streaker_offset']*1e6)), fontsize=config.fontsize)
            sp1.legend()
            sp2.legend()

    def reconstruct_current(self, tracker, gauss_kwargs, type_='centroid'):
        fit_dict = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]
        gap = fit_dict['gap_fit']
        streaker_offset = fit_dict['streaker_offset']
        gaps = [10e-3, 10e-3]
        gaps[self.n_streaker] = gap
        gauss_dicts = []
        offset_list = []

        for n_offset, offset in enumerate(self.offsets):
            if offset == 0:
                continue
            beam_offsets = [0., 0.]
            beam_offsets[self.n_streaker] = -(offset-streaker_offset)
            offset_list.append(beam_offsets[self.n_streaker])

            projx = self.images[n_offset].astype(np.float64).sum(axis=-2)
            median_proj = misc.get_median(projx)
            x_axis = self.plot_list_x[n_offset]
            if x_axis[1] < x_axis[0]:
                x_axis = x_axis[::-1]
                median_proj = median_proj[::-1]
            meas_screen = iap.ScreenDistribution(x_axis, median_proj, subtract_min=True)
            meas_screen.cutoff2(tracker.screen_cutoff)
            meas_screen.crop()
            meas_screen.reshape(tracker.len_screen)

            gauss_kwargs = gauss_kwargs.copy()
            gauss_kwargs['gaps'] = gaps
            gauss_kwargs['beam_offsets'] = beam_offsets
            gauss_kwargs['n_streaker'] = self.n_streaker
            gauss_kwargs['meas_screen'] = meas_screen

            #plt.figure()
            #plt.plot(meas_screen._xx, meas_screen._yy)
            #plt.show()
            #import pdb; pdb.set_trace()

            gauss_dict = tracker.find_best_gauss(**gauss_kwargs)
            gauss_dicts.append(gauss_dict)

        self.gauss_dicts_gap_order[type_][self.fit_gap][self.fit_order] = gauss_dicts
        return np.array(offset_list), gauss_dicts

    def plot_reconstruction(self, plot_handles=None):
        pass


def streaker_calibration_figure(figsize=[6.4, 4.8]):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Streaker center calibration')
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    subplot = ms.subplot_factory(2, 3)
    plot_handles = tuple((subplot(sp_ctr, title_fs=config.fontsize) for sp_ctr in range(1, 1+6)))
    clear_streaker_calibration(*plot_handles)
    return fig, plot_handles

def clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Centroid shift', 'Streaker center (mm)', 'Beam X centroid (mm)'),
            (sp_sizes, 'Size increase', 'Streaker center (mm)', 'Beam X rms (mm)'),
            (sp_center2, 'Centroid shift', 'Distance from jaw ($\mu$m)', 'Beam X centroid (mm)'),
            (sp_sizes2, 'Size increase', 'Distance from jaw ($\mu$m)', 'Beam X rms (mm)'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            (sp_current, 'Beam current', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title, fontsize=config.fontsize)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

