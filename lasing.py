import matplotlib.pyplot as plt
from collections import OrderedDict

try:
    from . import h5_storage
    from . import image_and_profile as iap
    from . import analysis
    from . import tracking
    from . import config
    from . import numpy as np
    from . import myplotstyle as ms
except ImportError:
    import h5_storage
    import image_and_profile as iap
    import analysis
    import tracking
    import config
    import numpy as np
    import myplotstyle as ms

class dummy_plot:
    def __init__(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        pass

    errorbar = plot
    legend = plot


def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
    power[power<0] = 0
    return power

def power_Eloss_err(slice_time, slice_current, slice_E_on, slice_E_off, slice_current_err, slice_E_on_err, slice_E_off_err):
    delta_E = slice_E_off-slice_E_on
    power = slice_current * delta_E
    power[power<0] = 0
    energy = np.trapz(power, slice_time)

    err_sq_1 = (delta_E * slice_current_err)**2
    err_sq_2 = (slice_current * slice_E_on_err)**2
    err_sq_3 = (slice_current * slice_E_off_err)**2
    power_err = np.sqrt(err_sq_1+err_sq_2+err_sq_3)

    return {
            'time': slice_time,
            'power': power,
            'energy': energy,
            'power_err': power_err,
            }

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total, norm_factor=None):
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    power0[power0<0] = 0
    integral = np.trapz(power0, slice_t)
    if norm_factor is None:
        power = power0/integral*E_total
    else:
        power = power0*norm_factor
    return power

def power_Espread_err(slice_t, slice_current, slice_Espread_on, slice_Espread_off, E_total, slice_current_err, slice_Espread_on_err, slice_Espread_off_err):

    slice_Espread_sqr_increase = slice_Espread_on**2 - slice_Espread_off**2
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    power0[power0 < 0] = 0
    integral = np.trapz(power0, slice_t)
    norm_factor = E_total/integral
    power = power0*norm_factor

    power0_err_1 = (slice_current**(-1/3) * slice_Espread_sqr_increase * slice_current_err)**2
    power0_err_2 = (slice_current**(2/3) * 2*slice_Espread_off * slice_Espread_off_err)**2
    power0_err_3 = (slice_current**(2/3) * 2*slice_Espread_on * slice_Espread_on_err)**2
    power0_err = np.sqrt(power0_err_1+power0_err_2+power0_err_3)

    power_err = power0_err*norm_factor
    energy = np.trapz(power, slice_t)

    return {
            'time': slice_t,
            'power': power,
            'power_err': power_err,
            'energy': energy,
            'norm_factor': norm_factor
            }

def obtain_lasing(image_off, image_on, n_slices, wake_x, wake_t, len_profile, dispersion, energy_eV, charge, pulse_energy, debug=False):

    all_slice_dict = OrderedDict()
    all_images = OrderedDict()

    for ctr, (image_obj, label) in enumerate([(image_off, 'Lasing_off'), (image_on, 'Lasing_on')]):

        image_cut = image_obj.cut(wake_x.min(), wake_x.max())
        image_reshaped = image_cut.reshape_x(len_profile)
        image_t = image_reshaped.x_to_t(wake_x, wake_t)
        if ctr == 0:
            ref_y = None
        image_tE, ref_y = image_t.y_to_eV(dispersion, energy_eV, ref_y)
        image_t_reduced = image_tE.slice_x(n_slices)
        slice_dict = image_t_reduced.fit_slice(charge=charge, smoothen_first=True, smoothen=1e6)
        all_slice_dict[label] = slice_dict
        all_images[label] = {
                'image_xy': image_obj,
                'image_tE': image_tE,
                'image_cut': image_cut,
                'image_t': image_t,
                'image_t_reduced': image_t_reduced,
                }

    slice_time = all_slice_dict['Lasing_off']['slice_x']
    mean_current = (all_slice_dict['Lasing_off']['slice_current']+all_slice_dict['Lasing_on']['slice_current'])/2.

    delta_E = all_slice_dict['Lasing_off']['slice_mean'] - all_slice_dict['Lasing_on']['slice_mean']
    delta_std_sq = all_slice_dict['Lasing_on']['slice_sigma']**2 - all_slice_dict['Lasing_off']['slice_sigma']**2
    np.clip(delta_std_sq, 0, None, out=delta_std_sq)

    power_from_Eloss = power_Eloss(mean_current, delta_E)
    E_total = np.trapz(power_from_Eloss, slice_time)
    power_from_Espread = power_Espread(slice_time, mean_current, delta_std_sq, pulse_energy)

    if debug:
        ms.figure('Lasing')
        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1
        sp_power = subplot(sp_ctr, title='Power')

        sp_ctr += 1
        sp_current = subplot(sp_ctr, title='Current')
        sp_ctr += 1
        sp_current.plot(slice_time, all_slice_dict['Lasing_off']['slice_current'], label='Off')
        sp_current.plot(slice_time, all_slice_dict['Lasing_on']['slice_current'], label='On')


        sp_power.plot(slice_time, power_from_Eloss)
        sp_power.plot(slice_time, power_from_Espread)
        plt.show()

    output = {
            'all_slice_dict': all_slice_dict,
            'power_Eloss': power_from_Eloss,
            'energy_Eloss': E_total,
            'power_Espread': power_from_Espread,
            'all_images': all_images,
            'current': mean_current,
            'slice_time': slice_time,
            }

    return output

def lasing_figure(figsize=None):
    fig = plt.figure(figsize=figsize)
    fig.canvas.set_window_title('Current reconstruction')
    fig.subplots_adjust(hspace=0.4)
    subplot = ms.subplot_factory(2,3)
    subplots = [subplot(sp_ctr) for sp_ctr in range(1, 1+5)]
    clear_lasing_figure(*subplots)
    return fig, subplots

def clear_lasing_figure(sp_slice_mean, sp_slice_sigma, sp_current, sp_lasing_loss, sp_lasing_spread):

    for sp, title, xlabel, ylabel in [
            (sp_slice_mean, 'Energy loss', 't (fs)','$\Delta E$ (MeV)'),
            (sp_slice_sigma, 'Energy spread increase', 't (fs)', 'Energy spread (MeV)'),
            (sp_current, 'Current profile', 't (fs)', 'Current (kA)'),
            (sp_lasing_loss, 'Energy loss power profile', 't (fs)', 'Power (GW)'),
            (sp_lasing_spread, 'Energy spread power profile', 't (fs)', 'Power (GW)'),
            ]:
        sp.clear()
        sp.set_title(title)
        sp.set_xlabel(xlabel)
        sp.set_ylabel(ylabel)
        sp.grid(True)


class LasingReconstruction:
    def __init__(self, images_off, images_on, pulse_energy=None, current_cutoff=1e3, key_mean='slice_cut_mean', key_sigma='slice_cut_rms'):
        assert images_off.profile == images_on.profile
        self.images_off = images_off
        self.images_on = images_on
        self.current_cutoff = current_cutoff
        self.pulse_energy = pulse_energy
        self.key_mean = key_mean
        self.key_sigma = key_sigma

        self.generate_all_slice_dict()
        self.lasing_analysis()

    def generate_all_slice_dict(self):
        self.all_slice_dict = {}
        for images, title, ls in [(self.images_off, 'Lasing Off', None), (self.images_on, 'Lasing On', '--')]:
            all_mean = np.zeros([len(images.images_tE), images.n_slices], dtype=float)
            all_sigma = all_mean.copy()
            all_x = all_mean.copy()
            all_current = all_mean.copy()

            for ctr, slice_dict in enumerate(images.slice_dicts):
                all_mean[ctr] = slice_dict[self.key_mean]
                all_sigma[ctr] = slice_dict[self.key_sigma]
                all_x[ctr] = slice_dict['slice_x']
                all_current[ctr] = slice_dict['slice_current']

            self.all_slice_dict[title] = {
                    'loss': all_mean,
                    'spread': all_sigma,
                    't': all_x,
                    'current': all_current,
                    }

        mean_slice_dict = self.mean_slice_dict = {}
        for title in 'Lasing Off', 'Lasing On':
            mean_slice_dict[title] = {}
            for key, arr in self.all_slice_dict[title].items():
                mean_slice_dict[title][key] = {
                        'mean': np.nanmean(arr, axis=0),
                        'std': np.nanstd(arr, axis=0),
                        }

    def lasing_analysis(self):
        all_slice_dict = self.all_slice_dict
        mean_slice_dict = self.mean_slice_dict
        self.lasing_dict = lasing_dict = {}

        mean_current = (mean_slice_dict['Lasing On']['current']['mean']+mean_slice_dict['Lasing Off']['current']['mean'])/2.
        self.current_mask = mask = np.abs(mean_current) > self.current_cutoff
        mean_current = mean_current[mask]
        err_current = (np.sqrt(mean_slice_dict['Lasing On']['current']['std']**2+mean_slice_dict['Lasing Off']['current']['std']**2)/2.)[mask]


        off_loss_mean = mean_slice_dict['Lasing Off']['loss']['mean'][mask]
        off_loss_err = mean_slice_dict['Lasing Off']['loss']['std'][mask]
        on_loss_mean = mean_slice_dict['Lasing On']['loss']['mean'][mask]
        on_loss_err = mean_slice_dict['Lasing On']['loss']['std'][mask]
        off_spread_mean = mean_slice_dict['Lasing Off']['spread']['mean'][mask]
        off_spread_err = mean_slice_dict['Lasing Off']['spread']['std'][mask]
        on_spread_mean = mean_slice_dict['Lasing On']['spread']['mean'][mask]
        on_spread_err = mean_slice_dict['Lasing On']['spread']['std'][mask]
        slice_time = mean_slice_dict['Lasing Off']['t']['mean'][mask]

        lasing_dict['time'] = slice_time
        lasing_dict['Eloss'] = power_Eloss_err(slice_time, mean_current, on_loss_mean, off_loss_mean, err_current, on_loss_err, off_loss_err)
        lasing_dict['Espread'] = power_Espread_err(slice_time, mean_current, on_spread_mean, off_spread_mean, self.pulse_energy, err_current, on_spread_err, off_spread_err)
        norm_factor = lasing_dict['Espread']['norm_factor']

        n_images = len(all_slice_dict['Lasing On']['t'])
        all_loss = np.zeros([n_images, mask.sum()])
        all_spread = all_loss.copy()

        for ctr in range(n_images):
            current = all_slice_dict['Lasing On']['current'][ctr, mask]
            mask2 = current < self.current_cutoff
            on_loss = all_slice_dict['Lasing On']['loss'][ctr,mask]
            on_spread = all_slice_dict['Lasing On']['spread'][ctr,mask]

            loss = off_loss_mean - on_loss
            power_loss = power_Eloss(mean_current, loss)
            power_loss[mask2] = 0
            all_loss[ctr] = power_loss

            sq_increase = on_spread**2 - off_spread_mean**2
            power_spread = power_Espread(slice_time, current, sq_increase, self.pulse_energy, norm_factor=norm_factor)
            power_spread[mask2] = 0
            all_spread[ctr] = power_spread
        lasing_dict['all_Eloss'] = all_loss
        lasing_dict['all_Espread'] = all_spread

    def plot(self, plot_loss=True, plot_spread=True, plot_handles=None):
        mask = self.current_mask

        if plot_handles is None:

            ms.figure('Slice properties')
            subplot = ms.subplot_factory(2,3)
            sp_ctr = 1
            if plot_loss:
                sp_slice_mean = subplot(sp_ctr, title='Energy loss', xlabel='t (fs)', ylabel='$\Delta E$ (MeV)')
                sp_ctr += 1
            else:
                sp_slice_mean = dummy_plot()
            if plot_spread:
                sp_slice_sigma = subplot(sp_ctr, title='Energy spread increase', xlabel='t (fs)', ylabel='Energy spread (MeV)')
                sp_ctr += 1
            else:
                sp_slice_sigma = dummy_plot()

            sp_current = subplot(sp_ctr, title='Current profile', xlabel='t (fs)', ylabel='Current (kA)')
            sp_ctr += 1

            if plot_loss:
                sp_lasing_loss = subplot(sp_ctr, title='Energy loss power profile', xlabel='t (fs)', ylabel='Power (GW)')
                sp_ctr += 1
            else:
                sp_lasing_loss = dummy_plot()
            if plot_spread:
                sp_lasing_spread = subplot(sp_ctr, title='Energy spread power profile', xlabel='t (fs)', ylabel='Power (GW)')
                sp_ctr += 1
            else:
                sp_lasing_spread = dummy_plot()
        else:
            sp_slice_mean, sp_slice_sigma, sp_current, sp_lasing_loss, sp_lasing_spread = plot_handles

        #print('Plotting...!')
        current_center = []
        for title, ls, mean_color in [('Lasing Off', None, 'black'), ('Lasing On', '--', 'red')]:
            all_slice_dict = self.all_slice_dict[title]
            mean_slice_dict = self.mean_slice_dict[title]
            for ctr in range(len(all_slice_dict['t'])):
                #color = ms.colorprog(ctr, len(all_slice_dict['x']))
                color = None
                xx_plot = all_slice_dict['t'][ctr,mask]
                sp_slice_mean.plot(xx_plot*1e15, all_slice_dict['loss'][ctr,mask]/1e6, color=color, ls=ls)
                sp_slice_sigma.plot(xx_plot*1e15, all_slice_dict['spread'][ctr,mask]/1e6, color=color, ls=ls)

            mean_mean = mean_slice_dict['loss']['mean'][mask]
            mean_std = mean_slice_dict['loss']['std'][mask]
            sigma_mean = mean_slice_dict['spread']['mean'][mask]
            sigma_std = mean_slice_dict['spread']['std'][mask]
            current_mean = mean_slice_dict['current']['mean']
            current_std = mean_slice_dict['current']['std']
            #import pdb; pdb.set_trace()
            sp_slice_mean.errorbar(xx_plot*1e15, mean_mean/1e6, yerr=mean_std/1e6, color=mean_color, ls=ls, lw=3, label=title)
            sp_slice_sigma.errorbar(xx_plot*1e15, sigma_mean/1e6, yerr=sigma_std/1e6, color=mean_color, ls=ls, lw=3, label=title)
            sp_current.errorbar(mean_slice_dict['t']['mean']*1e15, current_mean/1e3, yerr=current_std/1e3, label=title, color=mean_color)
            current_center.append(np.sum(mean_slice_dict['t']['mean']*current_mean)/current_mean.sum())

        self.images_off.profile.plot_standard(sp_current, center_float=np.mean(current_center), label='Reconstructed')
        sp_current.axhline(self.current_cutoff/1e3, color='black', ls='--')
        sp_current.legend()
        sp_slice_mean.legend()
        sp_slice_sigma.legend()

        lasing_dict = self.lasing_dict
        for key, label, sp in [
                ('Eloss', '$\Delta E$', sp_lasing_loss),
                ('Espread', r'$\Delta \langle E^2 \rangle$', sp_lasing_spread)]:
            xx_plot = lasing_dict[key]['time']*1e15
            yy_plot = lasing_dict[key]['power']/1e9
            yy_err = lasing_dict[key]['power_err']/1e9
            #sp.errorbar(xx_plot, yy_plot, yerr=yy_err, label=label, color='red', lw=3)
            yy_plot = np.nanmean(lasing_dict['all_'+key], axis=0)/1e9
            yy_err = np.nanstd(lasing_dict['all_'+key], axis=0)/1e9
            sp.errorbar(xx_plot, yy_plot, yerr=yy_err, label=label, color='black', lw=3)
        sp_lasing_loss.legend()
        sp_lasing_spread.legend()

        for key, label, sp in [('all_Eloss', '$\Delta E$', sp_lasing_loss), ('all_Espread', r'$\Delta \langle E^2 \rangle$', sp_lasing_spread)]:
            for y_arr in lasing_dict[key]:
                sp.plot(lasing_dict['time']*1e15, y_arr/1e9)


class LasingReconstructionImages:
    def __init__(self, n_slices, screen_x0, beamline, n_streaker, streaker_offset, gap, tracker_kwargs, profile=None, recon_kwargs=None, charge=None, subtract_median=False, noise_cut=0.1, max_rms=10e6):
        self.screen_x0 = screen_x0
        self.beamline = beamline
        self.n_streaker = n_streaker
        self.streaker_offset = streaker_offset
        self.n_slices = n_slices
        self.charge = charge
        self.gap = gap
        self.profile = profile
        self.recon_kwargs = recon_kwargs
        self.subtract_median = subtract_median
        self.noise_cut = noise_cut
        self.max_rms = max_rms

        self.tracker = tracking.Tracker(**tracker_kwargs)
        self.do_recon_plot = False

    def add_file(self, filename):
        data_dict = h5_storage.loadH5Recursive(filename)
        self.add_dict(data_dict)

    def add_dict(self, data_dict):
        meta_data = data_dict['meta_data_begin']
        self.tracker.set_simulator(meta_data)
        images = data_dict['pyscan_result']['image'].astype(float)
        x_axis = data_dict['pyscan_result']['x_axis_m'].astype(float)
        y_axis = data_dict['pyscan_result']['y_axis_m'].astype(float)
        self.add_images(meta_data, images, x_axis, y_axis)
        if self.charge is None:
            self.charge = meta_data[config.beamline_chargepv[self.beamline]]*1e-12
        if self.gap is None:
            self.gap = meta_data[config.streaker_names[self.beamline][self.n_streaker]+':GAP']*1e-3

    def add_images(self, meta_data, images, x_axis, y_axis):
        self.meta_data = meta_data
        self.x_axis0 = x_axis
        self.x_axis = x_axis - self.screen_x0
        self.y_axis = y_axis
        self.raw_images = images
        self.raw_image_objs = []
        self.meas_screens = []
        for n_image, img in enumerate(images):
            if self.subtract_median:
                img = img - np.median(img)
                img[img<0] = 0
            image = iap.Image(img, self.x_axis, y_axis)
            self.raw_image_objs.append(image)
            screen = iap.ScreenDistribution(image.x_axis, image.image.sum(axis=-2))
            self.meas_screens.append(screen)

    def get_current_profiles(self, blmeas_file=None):
        data_dict = {
                'meta_data_begin': self.meta_data,
                'pyscan_result': {
                    'image': self.raw_images,
                    'x_axis_m': self.x_axis0,
                    'y_axis_m': self.y_axis,
                    }
                }

        streaker_offsets = [0., 0.]
        streaker_offsets[self.n_streaker] = self.streaker_offset

        output_dicts = analysis.reconstruct_current(data_dict, self.n_streaker, self.beamline, self.tracker, 'All', self.recon_kwargs, self.screen_x0, streaker_offsets, blmeas_file, do_plot=self.do_recon_plot)
        self.profiles = [x['gauss_dict']['reconstructed_profile'] for x in output_dicts]
        for p in self.profiles:
            p._xx -= p._xx.min()

    def set_profile(self):
        rms = [x.rms() for x in self.profiles]
        index_median = np.argsort(rms)[len(rms)//2]
        self.profile = self.profiles[index_median]

    def calc_wake(self):
        streaker = config.streaker_names[self.beamline][self.n_streaker]
        beam_offset = -(self.meta_data[streaker+':CENTER']*1e-3 - self.streaker_offset)
        streaker_length = config.streaker_lengths[streaker]
        r12 = self.tracker.calcR12()[self.n_streaker]
        wake_t, wake_x = self.profile.get_x_t(self.gap, beam_offset, streaker_length, r12)
        self.wake_t, self.wake_x = wake_t, wake_x

    def cut_images(self):
        x_min, x_max = self.wake_x.min(), self.wake_x.max()
        self.cut_images = []
        for img in self.raw_image_objs:
            cut_img = img.cut(x_min, x_max)
            self.cut_images.append(cut_img)

    def convert_axes(self):
        dispersion = self.tracker.calcDisp()[self.n_streaker]
        self.dispersion = dispersion
        self.images_tE = []
        self.ref_y = []
        for img in self.cut_images:
            img_t = img.x_to_t(self.wake_x, self.wake_t)
            img_tE, ref_y = img_t.y_to_eV(dispersion, self.tracker.energy_eV)
            self.images_tE.append(img_tE)
            self.ref_y.append(ref_y)

    def slice_x(self):
        self.images_sliced = []
        for n_image, image in enumerate(self.images_tE):
            image_sliced = image.slice_x(self.n_slices)
            self.images_sliced.append(image_sliced)

    def fit_slice(self):
        self.slice_dicts = []
        for image in self.images_sliced:
            slice_dict = image.fit_slice(charge=self.charge, noise_cut=self.noise_cut, max_rms=self.max_rms)
            self.slice_dicts.append(slice_dict)

    def process_data(self):
        if self.profile is None:
            self.get_current_profiles()
            self.set_profile()
        self.calc_wake()
        self.cut_images()
        self.convert_axes()
        self.slice_x()
        self.fit_slice()

    def plot_images(self, type_, title='', **kwargs):
        if type_ == 'raw':
            images = self.raw_image_objs
        elif type_ == 'cut':
            images = self.cut_images
        elif type_ == 'tE':
            images = self.images_tE
        elif type_ == 'slice':
            images = self.images_sliced

        sp_ctr = np.inf
        ny, nx = 3, 3
        subplot = ms.subplot_factory(ny, nx, grid=False)

        for n_image, image in enumerate(images):
            if sp_ctr > ny*nx:
                ms.figure('%s Images %s' % (title, type_))
                sp_ctr = 1
            sp = subplot(sp_ctr, title='Image %i' % n_image, xlabel=image.xlabel, ylabel=image.ylabel)
            sp_ctr += 1
            slice_dict = None
            if type_ in ('tE', 'slice') and hasattr(self, 'slice_dicts'):
                slice_dict = self.slice_dicts[n_image]
            image.plot_img_and_proj(sp, slice_dict=slice_dict, **kwargs)

