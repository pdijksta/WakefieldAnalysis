from collections import OrderedDict
import pickle
import numpy as np
import matplotlib.pyplot as plt

import lasing
import gaussfit

import myplotstyle as ms

plt.close('all')

with open('./full_slice_dict.pkl', 'rb') as f:
    full_slice_dict = pickle.load(f)

compensate_jitter = True

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

limits = 42.5e-15, 75e-15

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

E_norm = energy_mean
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

