import numpy as np
import h5_storage
import image_and_profile as iap

import myplotstyle as ms

ms.closeall()

h5_file = '/afs/psi.ch/intranet/SF/Applications/WakefieldAnalysis/plots/068a_data.h5'


dict_ = h5_storage.loadH5Recursive(h5_file)


ms.figure('Comparison of profiles')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_dict = {}

for key, key2 in [('Positive', 'Streaker center negative'), ('Negative', 'Streaker center positive')]:
    sp = subplot(sp_ctr, grid=False, title=key2, xlabel='t (fs)', ylabel='I (kA)')
    sp_ctr += 1
    sp_dict[key] = sp


optics_keys = list(dict_.keys())

distance_pos = 278e-6
distance_neg = 253e-6

for optics_key in optics_keys:
    if optics_key in ('3', '9'):
        continue
    if optics_key not in ('1', '8'):
        continue

    if optics_key == '1':
        optics_key2 = '1'
    if optics_key == '8':
        optics_key2 = '7'

    rec_dict = dict_[optics_key]['reconstructions']
    beam_offsets_keys = list(rec_dict.keys())
    beam_offsets = np.array([float(x) for x in beam_offsets_keys])
    distances = dict_[optics_key]['gap']/2. - np.abs(beam_offsets)
    distances_pos = distances[beam_offsets > 0]
    distances_neg = distances[beam_offsets < 0]

    print(optics_key)
    print(distances_pos*1e6)
    print(distances_neg*1e6)

    for distance_arr, key, dist in [(distances_pos, 'Positive', distance_pos), (distances_neg, 'Negative', distance_neg)]:
        index = np.argmin((distance_arr - dist)**2).squeeze()
        if key == 'Positive':
            rec_key = str(beam_offsets[beam_offsets > 0][index])
        if key == 'Negative':
            rec_key = str(beam_offsets[beam_offsets < 0][index])

        bp = iap.BeamProfile.from_dict(rec_dict[rec_key]['reconstructed_profile'])
        bp.center()


        bp.plot_standard(sp_dict[key], label='%s - %i $\mu$m' % (optics_key2, distance_arr[index]*1e6))

for sp in sp_dict.values():
    sp.legend()





ms.show()

ms.saveall('/tmp/069_278', ending='.pdf')

