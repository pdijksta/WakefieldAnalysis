import matplotlib.pyplot as plt
import numpy as np
from socket import gethostname

import h5_storage
import analysis
import image_and_profile as iap

plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir.replace('18', '19')



files1 = [
        data_dir+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        data_dir+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5', # Affected but ok
        ]

files2 = [
        data_dir2+'2021_05_19-14_14_22_Calibration_SARUN18-UDCP020.h5',
        data_dir2+'2021_05_19-14_24_05_Calibration_SARUN18-UDCP020.h5',
        ]


file1, file2 = files1

dict1, dict2 = h5_storage.loadH5Recursive(file1), h5_storage.loadH5Recursive(file2)

beamline = 'Aramis'
fit_order = False
fit_gap = False
debug = False
forward_propagate_blmeas = False
force_screen_center = None
blmeas = None
tt_halfrange = 200e-15
charge = 200e-12
tracker = None
len_screen = 5000
magnet_data0 = dict1['raw_data']['meta_data_begin']
do_plot = True
plot_handles = None

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

for ctr, dict_ in enumerate([dict1, dict2]):
    meta_dict = dict_['meta_data']
    raw_dict = dict_['raw_data']
    where0 = np.argwhere(meta_dict['offsets'] == 0).squeeze()

    screen_x0 = meta_dict['centroid_mean'][where0]

    this_meta_data, this_plot_list, this_magnet_data0, this_raw_data = analysis.prepare_streaker_fit(raw_dict, None)

    this_meta_data['centroids'] -= screen_x0
    this_meta_data['centroid_mean'] -= screen_x0

    for key in 'semigap', 'streaker', 'screen':
        meta_data[key] = this_meta_data[key]

    mask_offset = this_meta_data['offsets'] != 0
    for key in 'centroids', 'centroid_mean', 'centroid_std', 'rms_mean', 'rms_std', 'offsets':
        if ctr == 0:
            meta_data[key].extend(this_meta_data[key][mask_offset])
        else:
            meta_data[key].extend(this_meta_data[key][::-1])

    plot_list.extend(this_plot_list)


for key in 'centroids', 'centroid_mean', 'centroid_std', 'rms_mean', 'rms_std', 'offsets':
    meta_data[key] = np.array(meta_data[key])

meta_data['beamline'] = beamline
offsets = meta_data['offsets']
centroid_mean = meta_data['centroid_mean']
centroid_std = meta_data['centroid_std']
semigap = meta_data['semigap']
rms_mean = meta_data['rms_mean']
rms_std = meta_data['rms_std']

fit_dict = analysis.two_sided_fit(offsets, centroid_mean, centroid_std, semigap, fit_order=fit_order, fit_gap=fit_gap, force_screen_center=force_screen_center, debug=debug)
fit_dict_rms = analysis.beamsize_fit(offsets, rms_mean, rms_std, semigap, fit_order=fit_order, fit_gap=fit_gap, debug=debug)

if fit_dict_rms is not None:
    meta_data['fit_dict_rms'] = fit_dict_rms
meta_data.update(fit_dict)

if forward_propagate_blmeas:
    blmeas_profile = iap.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV)
    blmeas_profile.cutoff2(5e-2)
    blmeas_profile.crop()
    blmeas_profile.reshape(len_screen)

    sim_screens = forward_propagate_blmeas(blmeas_profile, tt_halfrange, charge, tracker, magnet_data0, meta_data)
else:
    sim_screens = None
    blmeas_profile = None
meta_data['sim_screens'] = sim_screens
meta_data['blmeas_profile'] = blmeas_profile

if do_plot:
    analysis.plot_streaker_calib(meta_data, plot_handles, plot_list, forward_propagate_blmeas, len_screen)

output = {
        'meta_data': meta_data,
        }

plt.show()

