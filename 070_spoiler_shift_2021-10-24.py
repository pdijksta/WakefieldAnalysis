import os
import itertools
import lasing
import config
import h5_storage
import elegant_matrix

import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant')

directory = '/mnt/data/data_2021-10-24'



"""
https://elog-gfa.psi.ch/SwissFEL+commissioning/21994
Dijkstal, Bettoni, Vicario, Craievich, Arrell

     Obstacle monitor set at  578 m
    Streaker gap calibration
        /sf/data/measurements/2021/10/24/2021_10_24-14_11_59_Lasing_True_SARBD02-DSCR050.h5
        positive side: max  5.00
        negative side: max  4.25
        100 um in steps of 20 um
        863 um x_0, beamsize 31 um
        Delta gap -74 um
    FEL and current profile reconstruction: Anhang 1 (full beam lasing)
    14:14 - spoiling laser ON
        positive side (+5.00 streaker center)
            /sf/data/measurements/2021/10/24/2021_10_24-14_12_34_Lasing_False_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-14_11_59_Lasing_True_SARBD02-DSCR050.h5
            Anhang 2: Nice 2 colors
        negative side (-4.25 streaker center)
            /sf/data/measurements/2021/10/24/2021_10_24-14_16_14_Lasing_True_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-14_15_29_Lasing_False_SARBD02-DSCR050.h5
            Anhang 3: somewhat different results
    14:18 spoiling laser OFF
        negative side (-4.25 streaker center)
            /sf/data/measurements/2021/10/24/2021_10_24-14_18_52_Lasing_True_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-14_19_29_Lasing_False_SARBD02-DSCR050.h5
            Anhang 4 - FWHM 130 fs separation
        positive side (+5.00 streaker center)
            /sf/data/measurements/2021/10/24/2021_10_24-14_22_25_Lasing_False_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-14_23_09_Lasing_True_SARBD02-DSCR050.h5
            Anhang 5 - FWHM - somewhat larger
    repeat of calibration (lasing off this time)
        /sf/data/measurements/2021/10/24/2021_10_24-19_34_36_Calibration_SARUN18-UDCP020.h5
        -4.25 to -4.17 and +4.94 to + 5.02
        screen_x0 876 um, beamsize 29 um
        structure_center at 371 um
        gap delta -61 um
    19:43 spoiling laser ON
        negative side -4.25 mm
            /sf/data/measurements/2021/10/24/2021_10_24-19_41_28_Lasing_True_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-19_40_14_Lasing_False_SARBD02-DSCR050.h5
            Anhang 6
        positive side +5.00 mm
            /sf/data/measurements/2021/10/24/2021_10_24-19_48_55_Lasing_True_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-19_47_51_Lasing_False_SARBD02-DSCR050.h5
            Anhang 8
    19:46 spoiling laser OFF
        negative side -4.25 mm
            /sf/data/measurements/2021/10/24/2021_10_24-19_44_41_Lasing_False_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-19_43_50_Lasing_True_SARBD02-DSCR050.h5
            Anhang 7
        positive side +5.00 mm
            /sf/data/measurements/2021/10/24/2021_10_24-19_51_42_Lasing_True_SARBD02-DSCR050.h5
            /sf/data/measurements/2021/10/24/2021_10_24-19_52_31_Lasing_False_SARBD02-DSCR050.h5
            Anhang 9

"""

charge = 180e-12
delta_charge = 10e-12
beamline = 'Aramis'
n_streaker = 1


calib1 = (863e-6, 372e-6, -74e-6)
calib2 = (876e-6, 371e-6, -61e-6)

case1 = {
        1: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-14_12_34_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_11_59_Lasing_True_SARBD02-DSCR050.h5'
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-14_15_29_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_16_14_Lasing_True_SARBD02-DSCR050.h5',
                ],
            },
        0: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-14_19_29_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_18_52_Lasing_True_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-14_19_29_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_18_52_Lasing_True_SARBD02-DSCR050.h5',
                ],
            },
        }

case2 = {
        1: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_47_51_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_48_55_Lasing_True_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_40_14_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_41_28_Lasing_True_SARBD02-DSCR050.h5',
                ],
            },
        0: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_52_31_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_51_42_Lasing_True_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_44_41_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_43_50_Lasing_True_SARBD02-DSCR050.h5',
                ],
            },
        }

ene1 = 450e-6
ene2 = 530e-6


ms.closeall()

slice_factor = 3
current_cutoff = 0.3e3

for n_case, (calib, case, ene) in enumerate([
        (calib1, case1, ene1),
        (calib2, case2, ene2),
        ]):
    tracker_kwargs = config.get_default_tracker_settings()
    gauss_kwargs = config.get_default_gauss_recon_settings()

    for spoiler, direction in itertools.product([0, 1], [1, -1]):
        lasing_off_file, lasing_on_file = case[spoiler][direction]
        lasing_off_dict = h5_storage.loadH5Recursive(os.path.join(directory, os.path.basename(lasing_off_file)))
        lasing_on_dict = h5_storage.loadH5Recursive(os.path.join(directory, os.path.basename(lasing_on_file)))
        screen_x0, streaker_offset, delta_gap = calib

        las_rec_images = {}
        for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
            if spoiler == 1:
                gauss_kwargs['charge'] = charge + 10e-12
            else:
                gauss_kwargs['charge'] = charge
            rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=gauss_kwargs, charge=charge, subtract_median=True, slice_factor=slice_factor)

            rec_obj.add_dict(data_dict)
            if main_ctr == 1:
                rec_obj.profile = las_rec_images['Lasing Off'].profile
                rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
            rec_obj.process_data()
            las_rec_images[title] = rec_obj

        las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], ene, current_cutoff=current_cutoff)
        las_rec.plot(figsize=(14,10))

        ms.saveall('./las_rec/070_%i_%i_%i' % (n_case, spoiler, direction))
        ms.closeall()
        save_dict = {
                'all_slice_dict': las_rec.all_slice_dict,
                'mean_slice_dict': las_rec.mean_slice_dict,
                'lasing_dict': las_rec.lasing_dict,
                }
        h5_storage.saveH5Recursive('./las_rec/070_%i_%i_%i.h5' % (n_case, spoiler, direction), save_dict)


ms.show()

