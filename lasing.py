from collections import OrderedDict
import numpy as np

def power_Eloss(slice_current, slice_Eloss_eV):
    power = slice_current * slice_Eloss_eV
    return power

def power_Espread(slice_t, slice_current, slice_Espread_sqr_increase, E_total):
    power0 = slice_current**(2/3) * slice_Espread_sqr_increase
    integral = np.trapz(power0, slice_t)
    power = power0/integral*E_total
    return power

def obtain_lasing(image_off, image_on, n_slices, wake_x, wake_t, len_profile, dispersion, energy_eV, charge):

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
        slice_dict = image_t_reduced.fit_slice(charge=charge)
        all_slice_dict[label] = slice_dict
        all_images[label] = {
                'image_tE': image_tE,
                'image_cut': image_cut,
                'image_t': image_t,
                'image_t_reduced': image_t_reduced,
                }

    slice_time = all_slice_dict['Lasing_off']['slice_x']
    mean_current = (all_slice_dict['Lasing_off']['slice_current']+all_slice_dict['Lasing_on']['slice_mean'])/2.

    delta_E = all_slice_dict['Lasing_off']['slice_mean'] - all_slice_dict['Lasing_on']['slice_mean']
    delta_std_sq = all_slice_dict['Lasing_on']['slice_sigma']**2 - all_slice_dict['Lasing_off']['slice_sigma']**2

    power_from_Eloss = power_Eloss(mean_current, delta_E)
    E_total = np.trapz(power_from_Eloss, slice_time)
    power_from_Espread = power_Espread(slice_time, mean_current, delta_std_sq, E_total)

    output = {
            'all_slice_dict': all_slice_dict,
            'power_Eloss': power_from_Eloss,
            'energy_Eloss': E_total,
            'power_Espread': power_from_Espread,
            'all_images': all_images,
            }

    return output

