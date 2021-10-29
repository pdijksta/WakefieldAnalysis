import numpy as np
from socket import gethostname

import blmeas
import streaker_calibration
import image_and_profile as iap
import tracking
import config
import elegant_matrix
import myplotstyle as ms

ms.closeall()

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

charge = 180e-12
energy_eV = 6e9
gap0 = 10e-3

hostname = gethostname()
if hostname == 'desktop':
    data_dir1 = '/storage/data_2021-05-18/'
elif hostname == 'pc11292.psi.ch':
    data_dir1 = '/sf/data/measurements/2021/05/18/'
elif hostname == 'pubuntu':
    data_dir1 = '/mnt/data/data_2021-05-18/'

data_dir2 = data_dir1.replace('05', '10')
data_dir3 = data_dir2.replace('18', '24')

blmeas_file1 = data_dir1+'119325494_bunch_length_meas.h5'
blmeas_file2 = data_dir2+'132380133_bunch_length_meas.h5'
blmeas_file3 = data_dir3+'132879962_bunch_length_meas.h5'

sc_file1 = [data_dir1+'2021_05_18-23_07_20_Calibration_SARUN18-UDCP020.h5', data_dir1+'2021_05_18-23_32_12_Calibration_SARUN18-UDCP020.h5']
sc_file2 = [data_dir2+'2021_10_18-11_13_17_Calibration_data_SARUN18-UDCP020.h5']
sc_file3 = [data_dir3+'2021_10_24-10_34_00_Calibration_SARUN18-UDCP020.h5']


blmeas_files = [blmeas_file1, blmeas_file2, blmeas_file3]
all_sc_files = [sc_file1, sc_file2, sc_file3]
days = ['May 18', 'October 18', 'October 24']

fig = ms.figure('Bunch length meas')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_current = sp = subplot(sp_ctr, title='Bunch length measurements')
sp_ctr += 1

blmeas_dicts = []
blmeas_profiles = []


for ctr, (blmeas_file, day) in enumerate(zip(blmeas_files, days)):
    #if ctr == 0:
    #    profile = iap.profile_from_blmeas(blmeas_file, 200e-15, charge, energy_eV, True)
    #else:
    blmeas_dict = blmeas.load_avg_blmeas(blmeas_file)
    blmeas_dicts.append(blmeas_dict)
    profile = iap.BeamProfile(blmeas_dict[1]['time'][::-1], blmeas_dict[2]['current_reduced'][::-1], energy_eV, charge)
    profile.plot_standard(sp, label=day)
    blmeas_profiles.append(profile)

sp.legend()

gap_arr = np.array([gap0-130e-6, gap0+50e-6])

delta_gaps = []
streaker_offsets = []


min_max_distances = [
        (280e-6, 400e-6),
        (200e-6, 400e-6),
        (330e-6, 500e-6),
        ]


for sc_files, day, min_max_distance, blmeas_profile in zip(all_sc_files, days, min_max_distances, blmeas_profiles):
    sc = streaker_calibration.StreakerCalibration('Aramis', 1, gap0, charge)
    tracker_kwargs = config.get_default_tracker_settings()
    tracker = tracking.Tracker(**tracker_kwargs)
    recon_kwargs = config.get_default_gauss_recon_settings()
    recon_kwargs['charge'] = charge
    for sc_file in sc_files:
        sc.add_file(sc_file)
    tracker.set_simulator(sc.meta_data)
    fit_dict = sc.fit_type('centroid')
    streaker_offset = fit_dict['streaker_offset']
    streaker_offsets.append(streaker_offset)
    screen_x0 = sc.screen_x0_arr.mean()
    print('Streaker offset: %i' % (streaker_offset*1e6))
    print('Screen x0: %i' % (screen_x0*1e6))
    distance = gap0/2.-np.abs(sc.offsets-streaker_offset)
    print(distance*1e6)
    use_offsets = list(np.argwhere(np.logical_and(distance > min_max_distance[0], distance < min_max_distance[1])).squeeze())
    if not use_offsets:
        print('Continue for %s' % day)
        print(distance)
        continue
    #use_offsets = [1, 2, 3, 13, 14, 15]
    print(day, use_offsets)
    gap_recon_dict = sc.gap_reconstruction2(gap_arr, tracker, recon_kwargs, streaker_offset, gap0=gap0, use_offsets=use_offsets)
    sc.plot_gap_reconstruction(gap_recon_dict)
    ms.plt.suptitle(day)
    delta_gaps.append(gap_recon_dict['delta_gap'])
    print(delta_gaps[-1]*1e6)

    ms.plt.figure(fig.number)
    sp = subplot(sp_ctr, title=day, xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    blmeas_profile.cutoff2(tracker_kwargs['profile_cutoff'])
    blmeas_profile.crop()
    blmeas_profile.center()
    blmeas_profile.plot_standard(sp)

    gauss_dicts = gap_recon_dict['final_gauss_dicts']
    offset_list = gap_recon_dict['final_offset_list']

    for index in (0, 1, -2, -1):
        profile = gauss_dicts[index]['reconstructed_profile']
        if offset_list[index] > 0:
            label = 'pos %i' % (distance[index]*1e6)
        else:
            label = 'neg %i' % (distance[index]*1e6)
        profile.plot_standard(sp, label=label)

    sp.legend()


ms.show()

