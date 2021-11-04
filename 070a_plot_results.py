import numpy as np
import myplotstyle as ms
import h5_storage
import image_and_profile as iap

ms.closeall()

curr_fig = ms.figure('Current profiles')
subplot = ms.subplot_factory(2,2, grid=False)
curr_sp_ctr = 1




for n_case, label in [(0, '14:15'), (1, '19:45'),]:
    ms.plt.figure(curr_fig.number)
    sp_current = subplot(curr_sp_ctr, title='Current profiles %s' % label, xlabel='t (fs)', ylabel='I (kA)')
    curr_sp_ctr += 1
    ms.figure(label, figsize=(14,10))
    sp_ctr = 1
    for direction, direction_str in [(-1,'Negative'), (1, 'Positive')]:
        sp = subplot(sp_ctr, title='Streaking direction %s' % direction_str, xlabel='t (fs)', ylabel='P (GW)')
        sp_ctr += 1
        for spoiler, spoiler_str in [(0, 'Off'), (1, 'On')]:
            dict_ = h5_storage.loadH5Recursive('./las_rec/070_%i_%i_%i.h5' % (n_case, spoiler, direction))
            lasing_dict = dict_['lasing_dict']
            xx = lasing_dict['time']
            yy = lasing_dict['Espread']['power']
            xx -= np.sum(xx*yy)/np.sum(yy)
            yy_err = lasing_dict['Espread']['power_err']
            sp.errorbar(xx*1e15, yy/1e9, yerr=yy_err/1e9, label=spoiler_str)

            curr = dict_['mean_slice_dict']['Lasing Off']['current']['mean']
            curr_err = dict_['mean_slice_dict']['Lasing Off']['current']['std']
            curr_time = dict_['mean_slice_dict']['Lasing Off']['t']['mean']
            mean_t = np.sum(curr*curr_time)/np.sum(curr)
            sp_current.errorbar((curr_time-mean_t)*1e15, curr/1e3, yerr=curr_err/1e3, label='%s / %s' % (direction_str, spoiler_str))

            profile = iap.AnyProfile(curr_time, curr)
            profile.reshape(1000)
            print(label, direction_str, spoiler_str, 'FWHM %i fs' % round(profile.fwhm()*1e15))

        sp.legend(title='Spoiler laser')

    sp_current.legend(title='Direction / Spoiler')

    for spoiler, spoiler_str in [(0, 'Off'), (1, 'On')]:
        sp = subplot(sp_ctr, title='Spoiler laser %s' % spoiler_str, xlabel='t (fs)', ylabel='P (GW)')
        sp_ctr += 1
        for direction, direction_str in [(-1,'Negative'), (1, 'Positive')]:
            dict_ = h5_storage.loadH5Recursive('./las_rec/070_%i_%i_%i.h5' % (n_case, spoiler, direction))
            lasing_dict = dict_['lasing_dict']
            xx = lasing_dict['time']
            yy = lasing_dict['Espread']['power']
            xx -= np.sum(xx*yy)/np.sum(yy)
            yy_err = lasing_dict['Espread']['power_err']

            sp.errorbar(xx*1e15, yy/1e9, yerr=yy_err/1e9, label=direction_str)
        sp.legend(title='Streaking direction')



ms.saveall('./las_rec/070a', empty_suptitle=False, ending='.pdf')

ms.show()

