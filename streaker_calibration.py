import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

try:
    import h5_storage
    import config
    import image_and_profile as iap
    import myplotstyle as ms
    import misc2 as misc
except ImportError:
    from . import h5_storage
    from . import config
    from . import image_and_profile as iap
    from . import myplotstyle as ms
    from . import misc2 as misc


order0_centroid = 2.75
order0_rms = 2.70


class StreakerCalibration:
    proj_cutoff = 0.02

    def __init__(self, beamline, n_streaker, gap0, file_or_dict=None, offsets_range=None, images=None, x_axis=None, fit_gap=True, fit_order=False, order0_centroid=order0_centroid, order0_rms=order0_rms):
        self.order0_rms = order0_rms
        self.order0_centroid = order0_centroid
        self.fit_gap = fit_gap
        self.fit_order = fit_order
        self.gap0 = gap0
        self.n_streaker = n_streaker

        self.offsets = []
        self.centroids = []
        self.centroids_std = []
        self.rms = []
        self.rms_std = []
        self.images = None
        self.blmeas_profile = None
        self.sim_screen_dict = {}
        self.plot_list_x = []
        self.plot_list_y = []

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

        if offsets_range is not None:
            self.add_data(offsets_range, images, x_axis)
        if file_or_dict is not None:
            self.add_file(file_or_dict)

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
            self.images = np.concatenate([self.images, images[mask]])

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
        images = result_dict['image'].astype(float).squeeze()
        if 'x_axis_m' in result_dict:
            x_axis = result_dict['x_axis_m']
        elif 'x_axis' in result_dict:
            x_axis = result_dict['x_axis']
        else:
            print(result_dict.keys())
            raise KeyError

        offsets = data_dict['streaker_offsets'].squeeze()
        self.add_data(offsets, images, x_axis)

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
            order0 = order0_rms
        elif type_ == 'centroid':
            yy_mean = self.centroids
            yy_std = self.centroids_std
            fit_func = self.streaker_calibration_fit_func
            order0 = order0_centroid

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

    def forward_propagate(self, blmeas_profile, tt_halfrange, charge, tracker, type_='centroid'):
        tracker.set_simulator(self.meta_data)
        streaker_offset = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['streaker_offset']
        gap = self.fit_dicts_gap_order[type_][self.fit_gap][self.fit_order]['gap_fit']
        offsets = self.offsets
        len_screen = len(blmeas_profile._xx)
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
        return sim_screens

    def plot_streaker_calib(self, plot_handles=None):

        offsets = self.offsets
        fit_dict_centroid = self.fit_dicts_gap_order['centroid'][self.fit_gap][self.fit_order]
        fit_dict_rms = self.fit_dicts_gap_order['beamsize'][self.fit_gap][self.fit_order]

        xx_fit = fit_dict_centroid['xx_fit']
        xx_fit2 = fit_dict_centroid['xx_fit2']
        centroid_mean = self.centroids
        centroid_std = self.centroids_std
        screen_x0 = 0
        reconstruction = fit_dict_centroid['reconstruction']
        reconstruction2 = fit_dict_centroid['reconstruction2']
        rms_mean = self.rms
        rms_std = self.rms_std
        gap = fit_dict_centroid['gap_fit']
        fit_semigap = gap/2

        streaker_offset = fit_dict_centroid['streaker_offset']
        blmeas_profile = self.blmeas_profile
        forward_propagate_blmeas = (blmeas_profile is not None)
        if (gap, streaker_offset) in self.sim_screen_dict:
            sim_screens = self.sim_screen_dict[(gap, streaker_offset)]
            len_screen = len(sim_screens[0])
        else:
            sim_screens = None
            len_screen = int(1e3)

        meas_screens = []
        for n_proj, (x_axis, proj, offset) in enumerate(zip(self.plot_list_x, self.plot_list_y, offsets)):
            meas_screen = misc.proj_to_screen(proj, x_axis, False, screen_x0)
            meas_screen.cutoff2(3e-2)
            meas_screen.crop()
            meas_screen.reshape(len_screen)
            meas_screens.append(meas_screen)

        if plot_handles is None:
            fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = streaker_calibration_figure()
        else:
            (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = plot_handles
        rms_sim = np.zeros(len(offsets))
        centroid_sim = np.zeros(len(offsets))
        for n_proj, (meas_screen, offset) in enumerate(zip(meas_screens, offsets)):
            color = ms.colorprog(n_proj, offsets)
            meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
            if sim_screens is not None:
                sim_screen = sim_screens[n_proj]
                sim_screen.plot_standard(sp_proj, color=color, ls='--')
                centroid_sim[n_proj] = sim_screen.mean()
                rms_sim[n_proj] = sim_screen.rms()

        if forward_propagate_blmeas:
            blmeas_profile.plot_standard(sp_current, color='black')

        xx_plot = (offsets - streaker_offset)*1e3
        xx_plot_fit = (xx_fit - streaker_offset)*1e3
        sp_center.errorbar(xx_plot, (centroid_mean-screen_x0)*1e3, yerr=centroid_std*1e3, label='Data', ls='None', marker='o')
        sp_center.plot(xx_plot_fit, (reconstruction-screen_x0)*1e3, label='Fit')
        #initial_guess = meta_data['initial_guess']
        #sp_center.plot(xx_plot_fit, (initial_guess-screen_x0)*1e3, label='Guess')

        mask_pos, mask_neg = offsets > 0, offsets < 0
        xx_plot2 = np.abs(fit_semigap*1e3 - np.abs(xx_plot))
        for side_ctr, (mask2, label) in enumerate([(mask_pos, 'Positive'), (mask_neg, 'Negative')]):
            sp_center2.errorbar(xx_plot2[mask2], np.abs(centroid_mean[mask2]-screen_x0)*1e3, yerr=centroid_std[mask2]*1e3, label=label, marker='o', ls='None')

        plot2_sim = []
        for mask in mask_pos, mask_neg:
            plot2_sim.extend([(a, np.abs(b)*1e3) for a, b in zip(xx_plot2[mask], centroid_sim[mask])])
        plot2_sim.sort()
        xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
        sp_center2.plot(xx_plot_sim, yy_plot_sim, label='Simulated', ls='None', marker='o')

        xx_plot_fit2 = np.abs(fit_semigap - np.abs(xx_fit2 - streaker_offset))*1e3
        yy_plot_fit2 = (np.abs(reconstruction2)-screen_x0)*1e3
        xlims = sp_center2.get_xlim()
        mask_fit = np.logical_and(xx_plot_fit2 > xlims[0], xx_plot_fit2 < xlims[1])
        mask_fit = np.logical_and(mask_fit, xx_fit2 > 0)
        sp_center2.plot(xx_plot_fit2[mask_fit], yy_plot_fit2[mask_fit], label='Fit')
        sp_center2.set_xlim(*xlims)

        fit_semigap = fit_dict_rms['gap_fit']/2
        xx_plot = offsets - fit_dict_rms['streaker_offset']
        sp_sizes.errorbar(xx_plot*1e3, rms_mean*1e3, yerr=rms_std*1e3, label='Data', marker='o', ls='None')
        xx_plot_fit = (fit_dict_rms['xx_fit']-fit_dict_rms['streaker_offset'])*1e3
        try:
            sp_sizes.plot(xx_plot_fit, fit_dict_rms['reconstruction']*1e3, label='Fit')
        except:
            import pdb; pdb.set_trace()
        #sp_sizes.plot(xx_plot_fit, fit_dict_rms['initial_guess']*1e3, label='Guess')
        if sim_screens is not None:
            sp_sizes.plot(xx_plot*1e3, rms_sim*1e3, label='Simulated', marker='.', ls='None')

        plot2_sim = []
        for mask, label in [(offsets > 0, 'Positive'), (offsets < 0, 'Negative')]:
            xx_plot2 = np.abs(fit_semigap - np.abs(xx_plot))
            sp_sizes2.errorbar(xx_plot2[mask]*1e3, np.abs(rms_mean[mask])*1e3, yerr=rms_std[mask]*1e3, label=label, marker='o', ls='None')
            if sim_screens is not None:
                plot2_sim.extend([(a*1e3, b*1e3) for a, b in zip(xx_plot2[mask], rms_sim[mask])])

        if sim_screens is not None:
            plot2_sim.sort()
            xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
            sp_sizes2.plot(xx_plot_sim, yy_plot_sim, label='Simulated', ls='None', marker='o')


            xx_plot_fit2 = np.abs(fit_semigap - np.abs(fit_dict_rms['xx_fit2']-fit_dict_rms['streaker_offset']))*1e3
            xlims = sp_sizes2.get_xlim()
            mask_fit = np.logical_and(xx_plot_fit2 > xlims[0], xx_plot_fit2 < xlims[1])
            mask_fit = np.logical_and(mask_fit, fit_dict_rms['xx_fit2'] > 0)
            sp_sizes2.plot(xx_plot_fit2[mask_fit], fit_dict_rms['reconstruction2'][mask_fit]*1e3, label='Fit')
            sp_sizes.legend()
            sp_sizes2.legend()

        if sim_screens is not None:
            sp_center.plot(xx_plot*1e3, centroid_sim*1e3, label='Simulated', marker='.', ls='None')

        sp_center.legend()
        sp_center2.legend()


def streaker_calibration_figure():
    fig = plt.figure()
    fig.canvas.set_window_title('Streaker center calibration')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2, 3)
    plot_handles = tuple((subplot(sp_ctr) for sp_ctr in range(1, 1+6)))
    clear_streaker_calibration(*plot_handles)
    return fig, plot_handles

def clear_streaker_calibration(sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current):
    for sp, title, xlabel, ylabel in [
            (sp_center, 'Centroid shift', 'Streaker center (mm)', 'Beam X centroid (mm)'),
            (sp_sizes, 'Size increase', 'Streaker center (mm)', 'Beam X rms (mm)'),
            (sp_center2, 'Centroid shift', 'Distance from jaw (mm)', 'Beam X centroid (mm)'),
            (sp_sizes2, 'Size increase', 'Distance from jaw (mm)', 'Beam X rms (mm)'),
            (sp_proj, 'Screen projections', 'x (mm)', 'Intensity (arb. units)'),
            (sp_current, 'Beam current', 't (fs)', 'Current (kA)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(False)

def streaker_calibration_fit_func(offsets, streaker_offset, strength, order, const, semigap):
    c1 = np.abs((offsets-streaker_offset+semigap))**(-order)
    c2 = np.abs((offsets-streaker_offset-semigap))**(-order)
    return const + (c1 - c2)*strength

def streaker_calibration_fit_func_os(offsets, streaker_offset, strength, order, const, semigap):
    return const + np.abs((offsets-streaker_offset-semigap))**(-order)*strength

def one_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=True, fit_gap=False, force_screen_center=None, debug=False):
    if fit_gap:
        raise ValueError

    if force_screen_center is None:
        where0 = np.argwhere(offsets == 0).squeeze()
        const0 = centroid_mean[where0]
    else:
        const0 = force_screen_center
    argmax = np.argmax(np.abs(centroid_mean-const0)).squeeze()
    semigap *= np.sign(offsets[argmax])
    if abs(semigap) > abs(offsets[argmax]):
        offset0 = 0
    else:
        offset0 = offsets[argmax] - semigap - 200e-6*np.sign(semigap)
    order0 = order0_centroid

    s01 = (centroid_mean[argmax] - const0) * np.abs((offsets[argmax] - semigap - offset0))**order0
    p0 = [offset0, s01]
    if fit_order:
        p0.append(order0)

    def fit_func(*args):
        if fit_order:
            return streaker_calibration_fit_func_os(*args, const0, semigap)
        else:
            return streaker_calibration_fit_func_os(*args, order0, const0, semigap)
    try:
        p_opt, p_cov = curve_fit(fit_func, offsets, centroid_mean, p0, sigma=centroid_std)
    except RuntimeError:
        print('Streaker calibration did not converge')
        p_opt = p0
        if debug:
            fignum = plt.gcf().number
            plt.figure()
            plt.plot(offsets, centroid_mean, ls='None', marker='.')
            fit_xx = np.linspace(offsets[0], offsets[-1], 100)
            plt.plot(fit_xx, fit_func(fit_xx, *p0), ls='--')
            plt.show()
            plt.figure(fignum)
            import pdb; pdb.set_trace()

    streaker_offset = p_opt[0]
    if fit_gap:
        gap_fit = p_opt[-1]*2
    else:
        gap_fit = abs(semigap*2)
    if fit_order:
        order_fit = p_opt[2]
    else:
        order_fit = order0

    xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
    xx_fit2_min = -(gap_fit/2-streaker_offset-10e-6)
    xx_fit2_max = -xx_fit2_min + 2*streaker_offset
    xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
    reconstruction = fit_func(xx_fit, *p_opt)
    reconstruction2 = fit_func(xx_fit2, *p_opt)
    initial_guess = fit_func(xx_fit, *p0)
    return {
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
            'screen_x0': const0
            }

def beamsize_fit_func(offsets, streaker_offset, strength, order, semigap, const):
    sq0 = const**2
    c1 = np.abs((offsets-streaker_offset+semigap))**(-order*2)
    c2 = np.abs((offsets-streaker_offset-semigap))**(-order*2)
    if strength > 0:
        sq_add = strength * (c1+c2)
    else:
        sq_add = 0
    return np.sqrt(sq0 + sq_add)

def beamsize_fit(offsets, rms_mean, rms_std, semigap, fit_order=True, fit_gap=False, debug=False):

    where0 = np.argwhere(offsets == 0).squeeze()
    const0 = rms_mean[where0]
    order0 = order0_rms

    offset0 = (offsets[0] + offsets[-1])/2 + 20e-6

    s0_arr = (rms_mean**2 - const0**2) / (np.abs((offsets-offset0-semigap))**(-order0*2) + np.abs((offsets-offset0+semigap))**(-order0*2))
    s0 = (s0_arr[0] + s0_arr[-1])/2
    p0 = [offset0, s0]
    if fit_order:
        p0.append(order0)
    if fit_gap:
        p0.append(semigap)

    def fit_func(*args):
        global last_args
        last_args = args
        if fit_order:
            if fit_gap:
                return beamsize_fit_func(*args, const0)
            else:
                return beamsize_fit_func(*args, semigap, const0)
        else:
            if fit_gap:
                return beamsize_fit_func(*args[:-1], order0, args[-1], const0)
            else:
                return beamsize_fit_func(*args, order0, semigap, const0)

    add_on_y = np.ones(10)*const0
    add_on_x = np.linspace(-2e-3, 2e-3, 10)
    add_on_sig = np.ones(10)*rms_std[where0]
    offsets2 = np.concatenate([offsets[offsets < 0], add_on_x, offsets[offsets > 0]])
    rms_mean2 = np.concatenate([rms_mean[offsets < 0], add_on_y, rms_mean[offsets > 0]])
    rms_std2 = np.concatenate([rms_std[offsets < 0], add_on_sig, rms_std[offsets > 0]])

    try:
        p_opt, p_cov = curve_fit(fit_func, offsets2, rms_mean2, p0)
    except RuntimeError:
        print('Streaker calibration did not converge')
        p_opt = p0
        if debug:
            fignum = plt.gcf().number
            plt.figure()
            plt.errorbar(offsets2, rms_mean2, yerr=rms_std2, ls='None', marker='.')
            fit_xx = np.linspace(offsets[0], offsets[-1], 100)
            plt.plot(fit_xx, fit_func(fit_xx, *last_args[1:]), ls='--')
            plt.plot(fit_xx, fit_func(fit_xx, *p0), ls='--')
            plt.show()
            plt.figure(fignum)
            import pdb; pdb.set_trace()

    streaker_offset = p_opt[0]
    if fit_gap:
        gap_fit = p_opt[-1]*2
    else:
        gap_fit = abs(semigap*2)
    if fit_order:
        order_fit = p_opt[2]
    else:
        order_fit = order0

    xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
    xx_fit2_min = -(gap_fit/2-streaker_offset-10e-6)
    xx_fit2_max = -xx_fit2_min + 2*streaker_offset
    xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
    reconstruction = fit_func(xx_fit, *p_opt)
    reconstruction2 = fit_func(xx_fit2, *p_opt)
    initial_guess = fit_func(xx_fit, *p0)
    return {
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

def two_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=True, fit_gap=False, force_screen_center=None, debug=False):
    order0 = order0_centroid
    if force_screen_center is None:
        where0 = np.argwhere(offsets == 0).squeeze()
        const0 = centroid_mean[where0]
    else:
        const0 = force_screen_center
    delta_offset0 = (offsets.min() + offsets.max())/2
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

    streaker_offset = p_opt[0]
    if fit_gap:
        gap_fit = p_opt[-1]*2
    else:
        gap_fit = semigap*2

    if fit_order:
        order_fit = p_opt[2]
    else:
        order_fit = order0

    xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
    xx_fit2_min = -(gap_fit/2-streaker_offset-10e-6)
    xx_fit2_max = -xx_fit2_min + 2*streaker_offset
    xx_fit2 = np.linspace(xx_fit2_min, xx_fit2_max, int(1e3))
    reconstruction = fit_func(xx_fit, *p_opt)
    reconstruction2 = fit_func(xx_fit2, *p_opt)
    initial_guess = fit_func(xx_fit, *p0)

    return {
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
            'screen_x0': const0
            }

def prepare_streaker_fit(filename_or_dict, force_screen_center=None, force_gap=None):
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
                plot_list.append((x_axis, proj))
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
            plot_list.append((x_axis, proj))
        centroid_mean = centroids.squeeze()
        centroid_std = None
        rms_mean = rms
        rms_std = None

    streaker = data_dict['streaker']
    if 'meta_data_begin' in data_dict:
        key = 'meta_data_begin'
    else:
        key = 'meta_data'
    magnet_data0 = data_dict[key]
    if force_gap is not None:
        magnet_data0[streaker+':GAP'] = force_gap*1e3
    semigap = magnet_data0[streaker+':GAP']/2.*1e-3


    meta_data = {
        'centroids': centroids,
        'centroid_mean': centroid_mean,
        'centroid_std': centroid_std,
        'rms_mean': rms_mean,
        'rms_std': rms_std,
        'offsets': offsets,
        'semigap': semigap,
        'x_axis': x_axis,
        'streaker': streaker,
        'screen': data_dict['screen'],
        }
    return meta_data, plot_list, magnet_data0, data_dict


def analyze_streaker_calibration(filename_or_dict, do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None, forward_propagate_blmeas=False, tracker=None, blmeas=None, beamline='Aramis', charge=200e-12, fit_gap=False, debug=False, tt_halfrange=200e-15, len_screen=5000, force_gap=None):

    meta_data, plot_list, magnet_data0, raw_data = prepare_streaker_fit(filename_or_dict, force_screen_center, force_gap)
    meta_data['beamline'] = beamline
    offsets = meta_data['offsets']
    centroid_mean = meta_data['centroid_mean']
    centroid_std = meta_data['centroid_std']
    semigap = meta_data['semigap']
    rms_mean = meta_data['rms_mean']
    rms_std = meta_data['rms_std']

    for key, streaker2 in config.streaker_names[beamline].items():
        if meta_data['streaker'] == streaker2:
            n_streaker = key
            break
    else:
        raise ValueError('n_streaker not found')
    meta_data['n_streaker'] = n_streaker
    meta_data['beamline'] = beamline


    if offsets.min() == 0 or offsets.max() == 0:
        fit_dict = one_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=fit_order, fit_gap=fit_gap, force_screen_center=force_screen_center, debug=debug)
        fit_dict_rms = None
    else:
        fit_dict = two_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=fit_order, fit_gap=fit_gap, force_screen_center=force_screen_center, debug=debug)
        fit_dict_rms = beamsize_fit(offsets, rms_mean, rms_std, semigap, fit_order=fit_order, fit_gap=fit_gap, debug=debug)

    if fit_dict_rms is not None:
        meta_data['fit_dict_rms'] = fit_dict_rms
    meta_data.update(fit_dict)
    fit_semigap = meta_data['gap_fit']/2

    if forward_propagate_blmeas:
        blmeas_profile = iap.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV)
        blmeas_profile.cutoff2(3e-2)
        blmeas_profile.crop()
        blmeas_profile.reshape(len_screen)

        sim_screens = forward_propagate(blmeas_profile, tt_halfrange, charge, tracker, magnet_data0, meta_data, n_streaker, fit_semigap)
    else:
        sim_screens = None
        blmeas_profile = None
    meta_data['sim_screens'] = sim_screens
    meta_data['blmeas_profile'] = blmeas_profile

    if do_plot:
        plot_streaker_calib(meta_data, plot_handles, plot_list, forward_propagate_blmeas, len_screen)

    output = {
            'raw_data': raw_data,
            'meta_data': meta_data,
            }
    return output

def analyze_streaker_calibration_stitch_together(filename_or_dict1, filename_or_dict2, do_plot=True, plot_handles=None, fit_order=False, force_screen_center=None, forward_propagate_blmeas=False, tracker=None, blmeas=None, beamline='Aramis', charge=200e-12, fit_gap=False, debug=False, tt_halfrange=200e-15, len_screen=5000, force_gap=None):

    if type(filename_or_dict1) is dict:
        dict1 = filename_or_dict1
    else:
        dict1 = h5_storage.loadH5Recursive(filename_or_dict1)
    if type(filename_or_dict2) is dict:
        dict2 = filename_or_dict2
    else:
        dict2 = h5_storage.loadH5Recursive(filename_or_dict2)

    meta_data = {
        'centroids': [],
        'centroid_mean': [],
        'centroid_std': [],
        'rms_mean': [],
        'rms_std': [],
        'offsets': [],
        'semigap': None,
        'x_axis': None,
        'streaker': None,
        'screen': None,
        }

    plot_list = []

    diff1, diff2 = np.sign(np.diff(dict1['meta_data']['offsets'])[0]), np.sign(np.diff(dict2['meta_data']['offsets'])[0])

    for ctr, dict_ in enumerate([dict1, dict2]):
        meta_dict = dict_['meta_data']
        raw_dict = dict_['raw_data']
        where0 = np.argwhere(meta_dict['offsets'] == 0).squeeze()


        screen_x0 = meta_dict['centroid_mean'][where0]

        this_meta_data, this_plot_list, this_magnet_data0, this_raw_data = prepare_streaker_fit(raw_dict, None, force_gap)
        this_offsets = this_meta_data['offsets']

        if ctr == 0:
            reverse = np.abs(this_offsets[0]) < np.abs(this_offsets[-1])
            if reverse:
                diff1 *= -1
        else:
            reverse = diff1 != diff2

        this_meta_data['centroids'] -= screen_x0
        this_meta_data['centroid_mean'] -= screen_x0

        for key in 'semigap', 'streaker', 'screen':
            meta_data[key] = this_meta_data[key]

        if ctr == 0:
            mask_offset = this_offsets != 0
        else:
            mask_offset = np.ones_like(this_meta_data['offsets'], bool)

        for key in 'centroids', 'centroid_mean', 'centroid_std', 'rms_mean', 'rms_std', 'offsets':
            if reverse:
                meta_data[key].extend(this_meta_data[key][mask_offset][::-1])
            else:
                meta_data[key].extend(this_meta_data[key][mask_offset])

        if ctr == 0:
            where0 = np.argwhere(this_offsets == 0).squeeze()
            del this_plot_list[where0]

        if type(this_plot_list[0]) is tuple:
            if reverse:
                plot_list.extend([(x_axis-screen_x0, proj) for (x_axis, proj) in this_plot_list][::-1])
            else:
                plot_list.extend([(x_axis-screen_x0, proj) for (x_axis, proj) in this_plot_list])
        else:
            x_axis = raw_dict['pyscan_result']['x_axis_m']
            if reverse:
                plot_list.extend([(x_axis-screen_x0, proj) for proj in this_plot_list][::-1])
            else:
                plot_list.extend([(x_axis-screen_x0, proj) for proj in this_plot_list])

    for key in 'centroids', 'centroid_mean', 'centroid_std', 'rms_mean', 'rms_std', 'offsets':
        meta_data[key] = np.array(meta_data[key])

    meta_data['beamline'] = beamline
    offsets = meta_data['offsets']
    centroid_mean = meta_data['centroid_mean']
    centroid_std = meta_data['centroid_std']
    semigap = meta_data['semigap']
    rms_mean = meta_data['rms_mean']
    rms_std = meta_data['rms_std']

    for key, streaker2 in config.streaker_names[beamline].items():
        if meta_data['streaker'] == streaker2:
            n_streaker = key
            break
    else:
        raise ValueError('n_streaker not found', meta_data['streaker'], list(config.streaker_names[beamline].values()))
    meta_data['n_streaker'] = n_streaker
    meta_data['beamline'] = beamline

    fit_dict = two_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=fit_order, fit_gap=fit_gap, force_screen_center=force_screen_center, debug=debug)
    fit_dict_rms = beamsize_fit(offsets, rms_mean, rms_std, semigap, fit_order=fit_order, fit_gap=fit_gap, debug=debug)

    if fit_dict_rms is not None:
        meta_data['fit_dict_rms'] = fit_dict_rms
    meta_data.update(fit_dict)
    fit_semigap = meta_data['gap_fit']/2.

    if forward_propagate_blmeas:
        blmeas_profile = iap.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, subtract_min=True)
        blmeas_profile.cutoff2(3e-2)
        blmeas_profile.crop()
        blmeas_profile.reshape(len_screen)

        magnet_data0 = dict2['raw_data']['meta_data_begin']
        sim_screens = forward_propagate(blmeas_profile, tt_halfrange, charge, tracker, magnet_data0, meta_data, n_streaker, fit_semigap)
    else:
        sim_screens = None
        blmeas_profile = None
    meta_data['sim_screens'] = sim_screens
    meta_data['blmeas_profile'] = blmeas_profile

    if do_plot:
        plot_streaker_calib(meta_data, plot_handles, plot_list, forward_propagate_blmeas, len_screen)

    output = {
            'meta_data': meta_data,
            }
    return output

def forward_propagate(blmeas_profile, tt_halfrange, charge, tracker, magnet_data0, meta_data, n_streaker, fit_semigap):

    beamline = meta_data['beamline']
    streaker_name = meta_data['streaker']
    offsets = meta_data['offsets']
    streaker_offset = meta_data['streaker_offset']
    len_screen = len(blmeas_profile._xx)

    streaker_names = config.streaker_names[beamline]
    gaps = np.array([magnet_data0[x+':GAP']*1e-3 for x in streaker_names.values()])
    print('Gaps old (mm):', gaps*1e3)
    gaps[n_streaker] = fit_semigap*2
    print('Gaps new (mm):', gaps*1e3)
    streaker_offsets0 = []
    n_streaker_var = None
    for n_streaker, streaker in streaker_names.items():
        if streaker == streaker_name:
            n_streaker_var = n_streaker
            streaker_offsets0.append(None)
        else:
            streaker_offsets0.append(magnet_data0[streaker+':CENTER']*1e-3)
    assert n_streaker_var is not None

    sim_screens = []
    for s_offset in offsets:
        streaker_offsets = streaker_offsets0[:]
        streaker_offsets[n_streaker_var] = -(s_offset-streaker_offset)
        forward_dict = tracker.matrix_forward(blmeas_profile, gaps, streaker_offsets)
        sim_screen = forward_dict['screen']
        sim_screen.cutoff2(tracker.screen_cutoff)
        sim_screen.crop()
        sim_screen.reshape(len_screen)
        sim_screens.append(sim_screen)
    return sim_screens

def plot_streaker_calib(meta_data, plot_handles, plot_list, forward_propagate_blmeas, len_screen):

    offsets = meta_data['offsets']
    streaker_offset = meta_data['streaker_offset']
    xx_fit = meta_data['xx_fit']
    xx_fit2 = meta_data['xx_fit2']
    centroid_mean = meta_data['centroid_mean']
    centroid_std = meta_data['centroid_std']
    screen_x0 = meta_data['screen_x0']
    reconstruction = meta_data['reconstruction']
    reconstruction2 = meta_data['reconstruction2']
    rms_mean = meta_data['rms_mean']
    rms_std = meta_data['rms_std']
    sim_screens = meta_data['sim_screens']
    blmeas_profile = meta_data['blmeas_profile']
    fit_semigap = meta_data['gap_fit']/2

    meas_screens = []
    for n_proj, ((x_axis, proj), offset) in enumerate(zip(plot_list, offsets)):
        meas_screen = misc.proj_to_screen(proj, x_axis, False, screen_x0)
        meas_screen.cutoff2(3e-2)
        meas_screen.crop()
        meas_screen.reshape(len_screen)
        meas_screens.append(meas_screen)

    if plot_handles is None:
        fig, (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = streaker_calibration_figure()
    else:
        (sp_center, sp_sizes, sp_proj, sp_center2, sp_sizes2, sp_current) = plot_handles
    rms_sim = np.zeros(len(offsets))
    centroid_sim = np.zeros(len(offsets))
    for n_proj, (meas_screen, offset) in enumerate(zip(meas_screens, offsets)):
        color = ms.colorprog(n_proj, offsets)
        meas_screen.plot_standard(sp_proj, label='%.2f mm' % (offset*1e3), color=color)
        if sim_screens is not None:
            sim_screen = sim_screens[n_proj]
            sim_screen.plot_standard(sp_proj, color=color, ls='--')
            centroid_sim[n_proj] = sim_screen.mean()
            rms_sim[n_proj] = sim_screen.rms()

    if forward_propagate_blmeas:
        blmeas_profile.plot_standard(sp_current, color='black')

    xx_plot = (offsets - streaker_offset)*1e3
    xx_plot_fit = (xx_fit - streaker_offset)*1e3
    sp_center.errorbar(xx_plot, (centroid_mean-screen_x0)*1e3, yerr=centroid_std*1e3, label='Data', ls='None', marker='o')
    sp_center.plot(xx_plot_fit, (reconstruction-screen_x0)*1e3, label='Fit')
    #initial_guess = meta_data['initial_guess']
    #sp_center.plot(xx_plot_fit, (initial_guess-screen_x0)*1e3, label='Guess')

    mask_pos, mask_neg = offsets > 0, offsets < 0
    xx_plot2 = np.abs(fit_semigap*1e3 - np.abs(xx_plot))
    for side_ctr, (mask2, label) in enumerate([(mask_pos, 'Positive'), (mask_neg, 'Negative')]):
        sp_center2.errorbar(xx_plot2[mask2], np.abs(centroid_mean[mask2]-screen_x0)*1e3, yerr=centroid_std[mask2]*1e3, label=label, marker='o', ls='None')

    plot2_sim = []
    for mask in mask_pos, mask_neg:
        plot2_sim.extend([(a, np.abs(b)*1e3) for a, b in zip(xx_plot2[mask], centroid_sim[mask])])
    plot2_sim.sort()
    xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
    sp_center2.plot(xx_plot_sim, yy_plot_sim, label='Simulated', ls='None', marker='o')

    xx_plot_fit2 = np.abs(fit_semigap - np.abs(xx_fit2 - streaker_offset))*1e3
    yy_plot_fit2 = (np.abs(reconstruction2)-screen_x0)*1e3
    xlims = sp_center2.get_xlim()
    mask_fit = np.logical_and(xx_plot_fit2 > xlims[0], xx_plot_fit2 < xlims[1])
    mask_fit = np.logical_and(mask_fit, xx_fit2 > 0)
    sp_center2.plot(xx_plot_fit2[mask_fit], yy_plot_fit2[mask_fit], label='Fit')
    sp_center2.set_xlim(*xlims)

    if 'fit_dict_rms' in meta_data:
        fit_dict_rms = meta_data['fit_dict_rms']
        fit_semigap = fit_dict_rms['gap_fit']/2
        fit_dict_rms = meta_data['fit_dict_rms']
        xx_plot = offsets - fit_dict_rms['streaker_offset']
        sp_sizes.errorbar(xx_plot*1e3, rms_mean*1e3, yerr=rms_std*1e3, label='Data', marker='o', ls='None')
        xx_plot_fit = (fit_dict_rms['xx_fit']-fit_dict_rms['streaker_offset'])*1e3
        sp_sizes.plot(xx_plot_fit, fit_dict_rms['reconstruction']*1e3, label='Fit')
        #sp_sizes.plot(xx_plot_fit, fit_dict_rms['initial_guess']*1e3, label='Guess')
        if sim_screens is not None:
            sp_sizes.plot(xx_plot*1e3, rms_sim*1e3, label='Simulated', marker='.', ls='None')

        plot2_sim = []
        for mask, label in [(offsets > 0, 'Positive'), (offsets < 0, 'Negative')]:
            xx_plot2 = np.abs(fit_semigap - np.abs(xx_plot))
            sp_sizes2.errorbar(xx_plot2[mask]*1e3, np.abs(rms_mean[mask])*1e3, yerr=rms_std[mask]*1e3, label=label, marker='o', ls='None')
            if sim_screens is not None:
                plot2_sim.extend([(a*1e3, b*1e3) for a, b in zip(xx_plot2[mask], rms_sim[mask])])

        if sim_screens is not None:
            plot2_sim.sort()
            xx_plot_sim, yy_plot_sim = zip(*plot2_sim)
            sp_sizes2.plot(xx_plot_sim, yy_plot_sim, label='Simulated', ls='None', marker='o')


        xx_plot_fit2 = np.abs(fit_semigap - np.abs(fit_dict_rms['xx_fit2']-fit_dict_rms['streaker_offset']))*1e3
        xlims = sp_sizes2.get_xlim()
        mask_fit = np.logical_and(xx_plot_fit2 > xlims[0], xx_plot_fit2 < xlims[1])
        mask_fit = np.logical_and(mask_fit, fit_dict_rms['xx_fit2'] > 0)
        sp_sizes2.plot(xx_plot_fit2[mask_fit], fit_dict_rms['reconstruction2'][mask_fit]*1e3, label='Fit')
        sp_sizes.legend()
        sp_sizes2.legend()

    if sim_screens is not None:
        sp_center.plot(xx_plot*1e3, centroid_sim*1e3, label='Simulated', marker='.', ls='None')

    sp_center.legend()
    sp_center2.legend()

