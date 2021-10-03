import numpy as np
from h5_storage import loadH5Recursive
import config
import tracking
import streaker_calibration
import elegant_matrix
import h5_storage
import lasing
#import image_and_profile as iap

import myplotstyle as ms

elegant_matrix.set_tmp_dir('~/tmp_elegant')

ms.closeall()

dirname = '/sf/data/measurements/2021/10/03/'


#calib_file = dirname + '2021_10_03-13_23_01_Calibration_SARUN18-UDCP020.h5'
calib_file = dirname + '2021_10_03-14_17_05_Calibration_SARUN18-UDCP020.h5'

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
#sc.plot_list_x =[x-150e-6 for x in sc.plot_list_x]
fit_dict = sc.fit()[0]
streaker_offset = fit_dict['streaker_offset']



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


gap_arr = np.array([10e-3-100e-6, 10e-3+50e-6])
gap_reconstruction = sc.gap_reconstruction2(gap_arr, tracker, gauss_kwargs, streaker_offset)
sc.plot_gap_reconstruction(gap_reconstruction)


sc.reconstruct_current(tracker, gauss_kwargs, force_gap=gap_reconstruction['gap'], force_streaker_offset=streaker_offset, use_offsets=(0,1,2,-1,-2,-3))
sc.plot_reconstruction()

screen_x0 = fit_dict['screen_x0']
delta_gap = -(10e-3 - gap_reconstruction['gap'])
pulse_energy = 520e-6
slice_factor = 5
recon_kwargs = gauss_kwargs

file_on = '/sf/data/measurements/2021/10/03/2021_10_03-14_21_49_Lasing_True_SARBD02-DSCR050.h5'
file_off = '/sf/data/measurements/2021/10/03/2021_10_03-14_22_29_Lasing_False_SARBD02-DSCR050.h5'
lasing_off_dict = h5_storage.loadH5Recursive(file_off)
lasing_on_dict = h5_storage.loadH5Recursive(file_on)
las_rec_images = {}

for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
    rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True, slice_factor=slice_factor)

    rec_obj.add_dict(data_dict)
    if main_ctr == 1:
        rec_obj.profile = las_rec_images['Lasing Off'].profile
        rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
    rec_obj.process_data()
    las_rec_images[title] = rec_obj
    #rec_obj.plot_images('raw', title)
    #rec_obj.plot_images('tE', title)

las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=0.5e3)
las_rec.plot()








ms.show()



