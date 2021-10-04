import numpy as np
from h5_storage import loadH5Recursive, saveH5Recursive
import tracking
import config
import streaker_calibration
import elegant_matrix

import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

ms.closeall()


optics_list = [
        #n_optics, beta_x, alpha_x, beta_y, alpha_y
        (0, (5.53899, -0.706092, 17.5802, 1.90113)),
        (1, (4.94179, -0.573489, 16.7544, 1.81298)),
        (2, (5.19957, -0.411184, 15.2094, 1.04085,)),
        (3, (6.6936, -0.275315, 14.9284, 0.0938628,)),
        (5, (7.86378, -0.804572, 13.3418, 0.703483,)),
        (6, (10.2397, -1.18246, 12.1057, 0.488374,)),
        (7, (11.9012, -1.02114, 14.7195, 0.0616565,)),
        (8, (14.659, -1.34679, 14.1292, -0.0278493,)),
        (9, (14.8484, -1.16565, 10.6545, -0.294552,)),
        ]

files = [
        (0, '/sf/data/measurements/2021/10/03/2021_10_03-13_23_01_Calibration_SARUN18-UDCP020.h5',),
        (1, '/sf/data/measurements/2021/10/03/2021_10_03-13_56_36_Calibration_SARUN18-UDCP020.h5',),
        (2, '/sf/data/measurements/2021/10/03/2021_10_03-14_17_05_Calibration_SARUN18-UDCP020.h5',),
        (3, '/sf/data/measurements/2021/10/03/2021_10_03-14_38_51_Calibration_SARUN18-UDCP020.h5',),
        (5, '/sf/data/measurements/2021/10/03/2021_10_03-14_59_12_Calibration_SARUN18-UDCP020.h5',),
        (6, '/sf/data/measurements/2021/10/03/2021_10_03-15_06_51_Calibration_SARUN18-UDCP020.h5',),
        (7, '/sf/data/measurements/2021/10/03/2021_10_03-15_14_01_Calibration_SARUN18-UDCP020.h5',),
        (8, '/sf/data/measurements/2021/10/03/2021_10_03-15_26_11_Calibration_SARUN18-UDCP020.h5',),
        (9, '/sf/data/measurements/2021/10/03/2021_10_03-15_42_45_Calibration_SARUN18-UDCP020.h5',),
        ]


outp_dict = {}


for (n_optics0, optics), (n_optics1, file_) in zip(optics_list, files):

    assert n_optics0 == n_optics1


    calib_dict = loadH5Recursive(file_)['raw_data']

    charge=182e-12
    result_dict = calib_dict
    beamline = 'Aramis'
    force_gap = None
    tt_halfrange = None
    fit_gap = True
    fit_order = False
    forward_propagate_blmeas = False
    plot_handles = None

    meta_data = result_dict['meta_data_begin']
    streakers = list(config.streaker_names[beamline].values())
    offsets = np.array([meta_data[x+':CENTER'] for x in streakers])
    n_streaker = int(np.argmax(np.abs(offsets)).squeeze())

    if force_gap is None:
        gap0 = meta_data[streakers[n_streaker]+':GAP']*1e-3
    else:
        gap0 = force_gap
    if charge is None:
        charge = meta_data[config.beamline_chargepv[beamline]]*1e-12
    if tt_halfrange is None:
        tt_halfrange = config.get_default_gauss_recon_settings()['tt_halfrange']

    sc = streaker_calibration.StreakerCalibration(beamline, n_streaker, gap0, charge, file_or_dict=result_dict, fit_gap=fit_gap, fit_order=fit_order)
    #sc.screen_x0_arr = np.array(sc.screen_x0_arr) + 150e-6
    #sc.plot_list_x =[x-150e-6 for x in sc.plot_list_x]
    fit_dict = sc.fit()[0]
    streaker_offset = fit_dict['streaker_offset']

    sc.plot_streaker_calib(plot_handles)
    ms.plt.suptitle('Optics %i' % n_optics0)

    tracker_kwargs = config.get_default_tracker_settings()
    tracker_kwargs['optics0'] = (optics[0], optics[2], optics[1], optics[3],)
    tracker = tracking.Tracker(**tracker_kwargs)
    tracker.set_simulator(calib_dict['meta_data_begin'])

    gauss_kwargs = config.get_default_gauss_recon_settings()
    gauss_kwargs['charge'] = charge


    gap_arr = np.array([10e-3-100e-6, 10e-3+50e-6])
    gap_reconstruction = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_offset)
    sc.plot_gap_reconstruction(gap_reconstruction)
    ms.plt.suptitle('Optics %i' % n_optics0)


    offset_list, gauss_dicts = sc.reconstruct_current(tracker, gauss_kwargs, force_gap=gap_reconstruction['gap'], force_streaker_offset=streaker_offset, use_offsets=(0,1,2,-1,-2,-3))
    sc.plot_reconstruction()


    save_list = {offset: g for offset, g in zip(offset_list, gauss_dicts)}
    outp_dict[n_optics0] = {
            'streaker_center_fit': fit_dict,
            'reconstructions': save_list,
            }

    ms.saveall('./plots/068a_optics_%i' % n_optics0, empty_suptitle=False)
    ms.closeall()


saveH5Recursive('./plots/068a_data.h5', outp_dict)

ms.show()

