from socket import gethostname
import pickle
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np

from h5_storage import loadH5Recursive
import tracking
import lasing
import image_and_profile as iap
import misc2 as misc
import gaussfit
import myplotstyle as ms
import config

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

gap = 9931.87e-6
struct_offset = 5.05e-3
struct_center = 374e-6
screen_center = 0.906e-3
dispersion = 0.4394238
r12 = 7.13
n_streaker = 1

with open('./gauss_dicts.pkl', 'rb') as f:
    gauss_dicts = pickle.load(f)

final_profile = gauss_dicts[0]['reconstructed_profile']

energy_eV = final_profile.energy_eV
charge = 200e-12

gaps = [gap, gap]
beam_offsets = [0, -(struct_offset - struct_center)]
struct_lengths = [1, 1]

wake_t, wake_x = final_profile.get_x_t(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker], r12)
n_slices = 50
len_profile = int(2e3)
E_norm = 600e-6

#Lasing off
#lasing_on_file0 = '/sf/data/measurements/2021/05/18/2021_05_18-23_42_10_Lasing_True_SARBD02-DSCR050.h5'
#lasing_off_file0 = '/sf/data/measurements/2021/05/18/2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'

lasing_off_file0 = '/storage/data_2021-05-18/2021_05_18-20_05_21_Lasing_False_SARBD02-DSCR050.h5'
lasing_on_file0 = '/storage/data_2021-05-18/2021_05_18-20_04_28_Lasing_True_SARBD02-DSCR050.h5'


lasing_off_file = data_dir + os.path.basename(lasing_off_file0)
lasing_on_file = data_dir + os.path.basename(lasing_on_file0)

median_indices = OrderedDict()
median_images = OrderedDict()

full_slice_dict = OrderedDict()

for ctr, (data_file, label) in enumerate([(lasing_off_file, 'Lasing Off'), (lasing_on_file, 'Lasing On')]):
    data_dict = loadH5Recursive(data_file)
    tracker_kwargs = config.get_default_tracker_settings()
    tracker_kwargs['magnet_file'] = data_dict['meta_data_begin']
    tracker = tracking.Tracker(**tracker_kwargs)
    print('R12', tracker.calcR12()[1])
    print('disp', tracker.calcDisp()[1])
    images = data_dict['pyscan_result']['image'].astype(np.float64)
    x_axis = data_dict['pyscan_result']['x_axis_m'].astype(np.float64)
    y_axis = data_dict['pyscan_result']['y_axis_m'].astype(np.float64)

    projx = images.sum(axis=-2)

    median_index = misc.get_median(projx, output='index')
    median_indices[label] = median_index
    print(label, median_index)

    sp_ctr = np.inf
    ny, nx = 3, 3
    subplot = ms.subplot_factory(ny, nx)

    full_slice_dict[label] = np.zeros([len(images), 5, n_slices])

    if ctr == 1:
        ref_y = full_slice_dict['Lasing Off'][:,4,0].mean()

    for n_image, image in enumerate(images):
        image_obj = iap.Image(image, x_axis, y_axis, subtract_median=True, x_offset=screen_center)
        image_cut = image_obj.cut(wake_x.min(), wake_x.max())
        image_reshaped = image_cut.reshape_x(len_profile)
        image_t = image_reshaped.x_to_t(wake_x, wake_t, debug=False)
        if ctr == 0:
            ref_y = None
        image_tE, ref_y = image_t.y_to_eV(dispersion, energy_eV, ref_y)
        image_t_reduced = image_tE.slice_x(n_slices)
        slice_dict = image_t_reduced.fit_slice(charge=charge, smoothen_first=True, smoothen=1e6)
        full_slice_dict[label][n_image, 0] = slice_dict['slice_x']
        full_slice_dict[label][n_image, 1] = slice_dict['slice_mean']
        full_slice_dict[label][n_image, 2] = slice_dict['slice_sigma']
        full_slice_dict[label][n_image, 3] = slice_dict['slice_current']
        full_slice_dict[label][n_image, 4, 0] = ref_y


        if n_image == median_index:
            median_images[label] = image_obj

        if True:
            if sp_ctr > ny*nx:
                ms.figure('Images %s' % label)
                plt.subplots_adjust(hspace=0.35, wspace=0.3)
                sp_ctr = 1

            sp = subplot(sp_ctr, grid=False, title='Image %i' % n_image, xlabel='x [mm]', ylabel='y [mm]')
            sp_ctr += 1
            image_obj.plot_img_and_proj(sp)

#filename = './full_slice_dict.pkl'
#with open(filename, 'wb') as f:
#    pickle.dump(full_slice_dict, f)
#print('Saved %s' % filename)


image_on = median_images['Lasing On']
image_off = median_images['Lasing Off']
n_slices = 50
len_profile = int(2e3)
n_streaker=1

lasing_dict = lasing.obtain_lasing(image_off, image_on, n_slices, wake_x, wake_t, len_profile, dispersion, energy_eV, charge, 19e-6, debug=False)


ms.figure('Lasing analysis')
subplot = ms.subplot_factory(3,2)
sp_ctr = 1

image_dict = lasing_dict['all_images']
all_slice_dict = lasing_dict['all_slice_dict']

for label, subdict in image_dict.items():
    label2 = label.replace('_', ' ')

    #sp = subplot(sp_ctr, title=label2+' cut', xlabel='t [fs]', ylabel='$\Delta$E [MeV]')
    #sp_ctr += 1

    #image_cut = subdict['image_cut']
    #image_cut.plot_img_and_proj(sp)


    sp = subplot(sp_ctr, title=label2, xlabel='t [fs]', ylabel='$\Delta$E [MeV]', grid=False)
    sp_ctr += 1

    image_tE = subdict['image_tE']
    image_tE.plot_img_and_proj(sp)

    slice_dict = all_slice_dict[label]
    slice_x = slice_dict['slice_x']
    slice_mean = slice_dict['slice_mean']
    slice_sigma = slice_dict['slice_sigma']
    sp.errorbar(slice_x*1e15, slice_mean*1e-6, yerr=slice_sigma*1e-6, marker='_', color='red', ls='None')

sp = subplot(sp_ctr, title='Lasing', xlabel='t [fs]', ylabel='P [GW]')
sp_ctr += 1

slice_time = lasing_dict['slice_time']
power_Eloss = lasing_dict['power_Eloss']
power_Espread = lasing_dict['power_Espread']

sp.plot(slice_time, power_Eloss, color='red')
sp.plot(slice_time, power_Espread, color='blue')


compensate_jitter = False

figsize=(9, 7)


if compensate_jitter:
    index80 = np.argmin((full_slice_dict['Lasing On'][0,0] - 72.5e-15)**2)
    mean_ene = full_slice_dict['Lasing Off'][:,1,index80].mean()

    for label, arr in full_slice_dict.items():
        for n_image in range(len(arr)):
            arr[n_image,1] = arr[n_image,1] - arr[n_image,1,index80] + mean_ene

ms.figure('Slice analysis', figsize=figsize)
plt.subplots_adjust(hspace=.35)
subplot = ms.subplot_factory(2, 3, grid=False)
sp_ctr = 1

lasing_input = OrderedDict()

limits = 40.0e-15, 80e-15

for label, arr in full_slice_dict.items():

    sp_current = subplot(sp_ctr, title='Current %s' % label, xlabel='t [fs]', ylabel='I [kA]')
    sp_ctr += 1
    sp_current.set_ylim(0, 5)

    sp_mean = subplot(sp_ctr, title='E centroid %s' % label, xlabel='t [fs]', ylabel='$\Delta$E [MeV]')
    sp_ctr += 1
    sp_mean.set_ylim(-1, 3)

    sp_sigma = subplot(sp_ctr, title='E spread %s' % label, xlabel='t [fs]', ylabel='$\sigma_E$ [MeV]')
    sp_ctr += 1
    sp_sigma.set_ylim(0, 3)

    slice_time = arr[0, 0]

    avg_current = np.mean(arr[:,3], axis=0)
    mask = np.logical_and(slice_time > limits[0], slice_time < limits[1])
    slice_time2 = slice_time[mask]
    for n_image in range(len(arr)):
        slice_mean = arr[n_image, 1]
        slice_sigma = arr[n_image, 2]
        slice_current = arr[n_image, 3]

        sp_current.plot(slice_time*1e15, slice_current/1e3)
        sp_mean.plot(slice_time2*1e15, slice_mean[mask]/1e6)
        sp_sigma.plot(slice_time2*1e15, slice_sigma[mask]/1e6)

    for limit in limits:
        sp_current.axvline(limit*1e15, color='black', ls='--')



    mask = np.logical_and(slice_time > limits[0], slice_time < limits[1])
    slice_time2 = slice_time[mask]

    avg_mean = np.mean(arr[:,1][:,mask], axis=0)
    std_mean = np.std(arr[:,1][:,mask], axis=0)
    avg_sigma = np.mean(arr[:,2][:,mask], axis=0)
    std_sigma = np.std(arr[:,2][:,mask], axis=0)
    avg_current = np.mean(arr[:,3][:,mask], axis=0)
    std_current = np.std(arr[:,3][:,mask], axis=0)
    avg_current0 = np.mean(arr[:,3], axis=0)
    std_current0 = np.std(arr[:,3], axis=0)
    #sp_mean.plot(slice_time2*1e15, avg_mean/1e6, color='black', lw=3)
    #sp_sigma.plot(slice_time2*1e15, avg_sigma/1e6, color='black', lw=3)
    #sp_current.plot(slice_time2*1e15, avg_current/1e3, color='black', lw=3)

    lasing_input[label] = {
            'current0': avg_current0,
            'std_current0': std_current0,
            'current': avg_current,
            'std_current': std_current,
            'mean': avg_mean,
            'std_mean': std_mean,
            'sigma': avg_sigma,
            'std_sigma': std_sigma,
            }

ms.figure('Lasing analysis', figsize=(12, 7))
plt.subplots_adjust(hspace=.35)
subplot = ms.subplot_factory(2, 2, grid=False)
sp_ctr = 1

curr0 = (lasing_input['Lasing Off']['current0'] + lasing_input['Lasing On']['current0'])/2.
curr_err0 = np.sqrt((lasing_input['Lasing Off']['std_current0']/2)**2 +(lasing_input['Lasing On']['std_current0']/2)**2)
curr = curr0[mask]
curr_err = curr_err0[mask]


eloss = lasing_input['Lasing Off']['mean'] - lasing_input['Lasing On']['mean']
espread_sq = lasing_input['Lasing On']['sigma']**2 - lasing_input['Lasing Off']['sigma']**2

power_mean_dict = lasing.power_Eloss_err(
        slice_time2, curr,
        lasing_input['Lasing On']['mean'], lasing_input['Lasing Off']['mean'],
        curr_err, lasing_input['Lasing On']['std_mean'], lasing_input['Lasing Off']['std_mean'])
power_mean = power_mean_dict['power']
energy_mean = power_mean_dict['energy']
power_mean_err = power_mean_dict['power_err']

power_sigma_dict = lasing.power_Espread_err(
        slice_time2, curr,
        lasing_input['Lasing On']['sigma'], lasing_input['Lasing Off']['sigma'],
        energy_mean,
        curr_err, lasing_input['Lasing On']['std_sigma'], lasing_input['Lasing Off']['std_sigma'])
power_sigma = power_sigma_dict['power']
power_sigma_err = power_sigma_dict['power_err']

power_mean_err_sq = eloss * curr

sp_current = subplot(sp_ctr, title='Current', xlabel='t [fs]', ylabel='I [kA]')
sp_ctr += 1

for limit in limits:
    sp_current.axvline(limit*1e15, color='black', ls='--')

sp_mean = subplot(sp_ctr, title='Energy loss', xlabel='t [fs]', ylabel='$\Delta$E [MeV]')
sp_ctr += 1

sp_sigma = subplot(sp_ctr, title='Energy spread', xlabel='t [fs]', ylabel='$\sigma_E$ [MeV]')
sp_ctr += 1

for label, subdict in lasing_input.items():
    sp_current.errorbar(slice_time*1e15, subdict['current0']/1e3, yerr=subdict['std_current0']/1e3, marker='.', label=label)
    sp_mean.errorbar(slice_time2*1e15, subdict['mean']/1e6, yerr=subdict['std_mean']/1e6, marker='.', label=label)
    sp_sigma.errorbar(slice_time2*1e15, subdict['sigma']/1e6, yerr=subdict['std_sigma']/1e6, marker='.', label=label)

sp_current.errorbar(slice_time*1e15, curr0/1e3, yerr=curr_err0/1e3, label='Average', color='black')

sp_lasing = subplot(sp_ctr, title='FEL power (integrated power is %i uJ)' % (energy_mean*1e6), xlabel='t [fs]', ylabel='P [GW]')
sp_ctr += 1

# Gas detector reading 19 uJ


for sp_ in sp_current, sp_mean, sp_sigma:
    sp_.legend()

ms.figure('Single compared to average', figsize=(12, 7))
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_mean = subplot(sp_ctr, title='$P_m$', xlabel='t [fs]', ylabel='P [GW]')
sp_ctr += 1

#E_norm = energy_mean
sp_sigma = subplot(sp_ctr, title='$P_\sigma$ (normalized to %i uJ)' % (E_norm*1e6), xlabel='t [fs]', ylabel='P [GW]')
sp_ctr += 1


arr = full_slice_dict['Lasing On']
for n_image in range(len(arr)):
    slice_mean = arr[n_image, 1, mask]
    slice_sigma = arr[n_image, 2, mask]
    slice_eloss = lasing_input['Lasing Off']['mean'] - slice_mean
    slice_Espread_sqr_increase = slice_sigma**2 - lasing_input['Lasing Off']['sigma']**2

    p_eloss = lasing.power_Eloss(curr, slice_eloss)
    E_eloss = np.trapz(p_eloss, slice_time2)
    p_espread = lasing.power_Espread(slice_time2, curr, slice_Espread_sqr_increase, E_norm)

    sp_mean.plot(slice_time2*1e15, p_eloss/1e9)
    sp_sigma.plot(slice_time2*1e15, p_espread/1e9)

for ctr, (power, error, label, color, sp2) in enumerate([
        (power_mean, power_mean_err, '$P_m$', 'red', sp_mean),
        (power_sigma, power_sigma_err, '$P_\sigma$', 'blue', sp_sigma),
        ]):

    gf = gaussfit.GaussFit(slice_time2, power, fit_const=False)
    label2 = label+ ' $\sigma$=%.1f fs' % (gf.sigma*1e15)
    for sp_, color2, lw in [(sp_lasing, color, None), (sp2, 'black', 3)]:
        sp_.errorbar(slice_time2*1e15, power/1e9, yerr=error/1e9, label=label2, color=color2, lw=lw)

sp_lasing.legend()



ms.saveall('/tmp/lasing_analysis', ending='.pdf', hspace=.35, wspace=.35)


plt.show()



plt.show()

