import numpy as np
from h5_storage import loadH5Recursive
import config
import tracking
import streaker_calibration
import elegant_matrix
#import image_and_profile as iap

import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant')

ms.closeall()


calib_file = '/sf/data/measurements/2021/10/03/2021_10_03-10_32_26_Calibration_SARUN18-UDCP020.h5'

calib_dict = loadH5Recursive(calib_file)['raw_data']

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
sc.plot_list_x =[x-150e-6 for x in sc.plot_list_x]
sc.fit()
#if forward_propagate_blmeas:
#    beam_profile = iap.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, True, 1)
#    beam_profile.reshape(tracker.len_screen)
#    beam_profile.cutoff2(5e-2)
#    beam_profile.crop()
#    beam_profile.reshape(tracker.len_screen)
#    sc.forward_propagate(beam_profile, tt_halfrange, tracker)
sc.plot_streaker_calib(plot_handles)



tracker_kwargs = config.get_default_tracker_settings()
tracker = tracking.Tracker(**tracker_kwargs)
tracker.set_simulator(calib_dict['meta_data_begin'])

gauss_kwargs = config.get_default_gauss_recon_settings()
gauss_kwargs['charge'] = charge

sc.reconstruct_current(tracker, gauss_kwargs)
sc.plot_reconstruction()







ms.show()



