
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

charge = 180e-6
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
                '/sf/data/measurements/2021/10/24/2021_10_24-14_18_52_Lasing_True_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_19_29_Lasing_False_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-14_18_52_Lasing_True_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-14_19_29_Lasing_False_SARBD02-DSCR050.h5',
                ],
            },
        }

case2 = {
        1: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_48_55_Lasing_True_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_47_51_Lasing_False_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_41_28_Lasing_True_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_40_14_Lasing_False_SARBD02-DSCR050.h5',
                ],
            },
        0: {
            1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_51_42_Lasing_True_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_52_31_Lasing_False_SARBD02-DSCR050.h5',
                ],
            -1: [
                '/sf/data/measurements/2021/10/24/2021_10_24-19_44_41_Lasing_False_SARBD02-DSCR050.h5',
                '/sf/data/measurements/2021/10/24/2021_10_24-19_43_50_Lasing_True_SARBD02-DSCR050.h5',
                ],
            },
        }

ene1 = 450e-6
ene2 = 530e-6





