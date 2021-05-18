from collections import OrderedDict
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
        image_t = image_reshaped.x_to_t(wake_x, wake_t, debug=False)
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

