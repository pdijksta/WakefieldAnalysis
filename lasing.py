from collections import OrderedDict
import h5_storage
import image_and_profile as iap
import analysis
import tracking
import config
import numpy as np

try:
    #from h5_storage import loadH5Recursive
    #import misc2 as misc
    pass
except ImportError:
    pass
    #from WakefieldAnalysis.h5_storage import loadH5Recursive
    #import WakefieldAnalysis.misc2 as misc

def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
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
            'power': power,
            'energy': energy,
            'power_err': power_err,
            }

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total):
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    integral = np.trapz(power0, slice_t)
    power = power0/integral*E_total
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
            'power': power,
            'power_err': power_err,
            'energy': energy,
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
        import matplotlib.pyplot as plt
        import myplotstyle as ms
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
        #import pdb; pdb.set_trace()


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

class LasingReconstruction:
    def __init__(self):
        pass

class LasingReconstructionImages:
    def __init__(self, screen_x0, beamline, n_streaker, streaker_offset, gap, tracker_kwargs, profile=None):
        self.screen_x0 = screen_x0
        self.beamline = beamline
        self.n_streaker = n_streaker
        self.streaker_offset = streaker_offset
        self.gap = gap
        self.profile = profile
        self.profiles = None
        self.tracker = tracking.Tracker(**tracker_kwargs)

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

    def add_images(self, meta_data, images, x_axis, y_axis):
        self.meta_data = meta_data
        self.x_axis0 = x_axis
        self.x_axis = x_axis - self.screen_x0
        self.y_axis = y_axis
        self.raw_images = images
        self.raw_image_objs = []
        self.meas_screens = []
        for n_image, img in enumerate(images):
            image = iap.Image(img, self.x_axis, y_axis)
            self.raw_image_objs.append(image)
            screen = iap.ScreenDistribution(image.x_axis, image.image.sum(axis=-2))
            self.meas_screens.append(screen)

    def get_current_profiles(self, kwargs_recon, blmeas_file=None, do_plot=False):
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

        output_dicts = analysis.reconstruct_current(data_dict, self.n_streaker, self.beamline, self.tracker, 'All', kwargs_recon, self.screen_x0, streaker_offsets, blmeas_file, do_plot=do_plot)
        self.profiles = [x['gauss_dict']['reconstructed_profile'] for x in output_dicts]

    def set_median_profile(self):
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
        self.images_tE = []
        for img in self.cut_images:
            img_t = img.x_to_t(self.wake_x, self.wake_t)
            img_tE = img_t.y_to_eV(dispersion, self.tracker.energy_eV)
            self.images_tE.append(img_tE)


