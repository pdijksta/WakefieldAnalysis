import numpy as np
import myplotstyle as ms
import h5_storage

ms.closeall()

for n_case, label in [(0, '14:15'), (1, '19:45'),]:
    ms.figure(label, figsize=(14,10))
    subplot = ms.subplot_factory(2, 2, grid=False)
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
        sp.legend(title='Spoiler laser')

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


ms.saveall('./las_rec/070a')

ms.show()

