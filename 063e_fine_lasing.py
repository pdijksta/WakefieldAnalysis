#import sys
import numpy as np
import argparse
from socket import gethostname

#import streaker_calibration
import h5_storage
import elegant_matrix
import lasing
import config
#import tracking
#import analysis
import myplotstyle as ms

parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--save', type=str)
args = parser.parse_args()



elegant_matrix.set_tmp_dir('~/tmp_elegant/')

ms.closeall()

hostname = gethostname()
if hostname == 'desktop':
    data_dir2 = '/storage/data_2021-05-19/'
elif hostname == 'pc11292.psi.ch':
    data_dir2 = '/sf/data/measurements/2021/05/19/'
elif hostname == 'pubuntu':
    data_dir2 = '/mnt/data/data_2021-05-19/'
data_dir1 = data_dir2.replace('19', '18')


sc_file = data_dir1+'2021_05_18-22_11_36_Calibration_SARUN18-UDCP020.h5'
# Full lasing, but saturation
lasing_on_fileF = data_dir1+'2021_05_18-23_42_10_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileF = data_dir1+'2021_05_18-23_43_39_Lasing_False_SARBD02-DSCR050.h5'

# Full lasing begin
lasing_off_fileFB = data_dir1+'2021_05_18-21_02_13_Lasing_False_SARBD02-DSCR050.h5'
lasing_on_fileFB = data_dir1+'2021_05_18-20_52_52_Lasing_True_SARBD02-DSCR050.h5'

# Short pulse begin
lasing_on_fileSB = data_dir1+'2021_05_18-21_08_24_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileSB = data_dir1+'2021_05_18-21_06_46_Lasing_False_SARBD02-DSCR050.h5'


# Short pulse
lasing_on_fileS = data_dir1+'2021_05_18-23_47_11_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_fileS = data_dir1+'2021_05_18-23_48_12_Lasing_False_SARBD02-DSCR050.h5'

#Two color pulse I=3 A, k=2
lasing_on_file2 = data_dir1+'2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
lasing_off_file2 = data_dir1+'2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'

blmeas_file = data_dir1+'119325494_bunch_length_meas.h5'

screen_x00 = 4250e-6
screen_x02 = 898.02e-6

streaker_offset0 = 374e-6
streaker_offset2 = 364e-6

main_fig = ms.figure('Main lasing', figsize=(8, 6))
hspace, wspace = 0.4, 0.4
ms.plt.subplots_adjust(hspace=hspace, wspace=wspace)
subplot = ms.subplot_factory(2, 3, grid=False)
sp_ctr = 1

rec_ctr = 2


norm_factor = None
for ctr, (lasing_on_file, lasing_off_file, pulse_energy, screen_x0, streaker_offset, curr_lim, main_title) in enumerate([
        (lasing_on_fileFB, lasing_off_fileFB, 625e-6, screen_x02, streaker_offset2, 1e3, 'Max pulse energy'),
        (lasing_on_file2, lasing_off_file2, 180e-6, screen_x02, streaker_offset2, 1e3, 'Two-color'),
        (lasing_on_fileSB, lasing_off_fileSB, 85e-6, screen_x02, streaker_offset2, 1.5e3, 'Short pulse'),
        ]):

    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)

    n_streaker = 1
    beamline = 'Aramis'
    delta_gap = -62e-6
    tracker_kwargs = config.get_default_tracker_settings()
    recon_kwargs = config.get_default_gauss_recon_settings()
    slice_factor = 3
    charge = 200e-12

    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True, slice_factor=slice_factor)
        #if ctr == rec_ctr:
        #    rec_obj.do_recon_plot = True

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
            rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
            rec_obj.ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        rec_obj.process_data()
        las_rec_images[title] = rec_obj
        #rec_obj.plot_images('raw', title)
        #rec_obj.plot_images('tE', title)

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=curr_lim, norm_factor=norm_factor)
    las_rec.plot()
    if ctr == 0:
        norm_factor = las_rec.norm_factor
    if ctr == rec_ctr:
        lasing_off_dict_fb = lasing_off_dict
        las_rec_fb = las_rec
        rec_obj_fb = las_rec_images['Lasing Off']


    ms.plt.figure(main_fig.number)
    sp = subplot(sp_ctr, title=main_title, xlabel='t (fs)', ylabel='$\Delta$E (MeV)', grid=False)
    sp_ctr += 1

    sp_espread = subplot(sp_ctr+2, xlabel='t (fs)', ylabel='P (GW)')
    sp_dummy = lasing.dummy_plot()

    las_rec.plot(plot_handles=(sp_dummy, sp_dummy, sp_dummy, sp_dummy, sp_espread, sp_dummy))
    sp_espread.get_legend().remove()

    rec_obj = las_rec_images['Lasing On']
    image_tE = rec_obj.images_tE[0]
    slice_dict = rec_obj.slice_dicts[0]
    image_tE.plot_img_and_proj(sp, plot_gauss=False, ylim=[-8e6, 6e6], slice_dict=slice_dict)

#ms.figure('Current profile reconstruction')
#subplot = ms.subplot_factory(2,3)
#
#sc_dict = h5_storage.loadH5Recursive(sc_file)
#sc = streaker_calibration.StreakerCalibration('Aramis', n_streaker, 10e-3, sc_dict)
#
#
#sp_screen, sp_profile, sp_moments = [subplot(x+1) for x in range(3)]
#sp_opt = lasing.dummy_plot()
#
#for sp, title, xlabel, ylabel in [
#        (sp_screen, 'Screen', 'x (mm)', 'Intensity (arb. units)'),
#        (sp_profile, 'Profile', 't (fs)', 'Current (kA)'),
#        (sp_opt, 'Optimization', 'Gaussian $\sigma$ (fs)', 'Opt value'),
#        (sp_moments, 'Moments', 'Gaussian $\sigma$ (fs)', r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)'),
#        ]:
#    sp.clear()
#    sp.set_title(title)
#    sp.set_xlabel(xlabel)
#    sp.set_ylabel(ylabel)
#    sp.grid(True)
#
#plot_handles = sp_screen, sp_profile, sp_opt, sp_moments
#
#
#
#tracker_kwargs = config.get_default_tracker_settings()
#recon_kwargs = config.get_default_gauss_recon_settings()
#tracker = tracking.Tracker(**tracker_kwargs)
#tracker.set_simulator(lasing_off_dict_fb['meta_data_begin'])
#
#index = rec_obj_fb.median_meas_screen_index
#meas_screen = rec_obj_fb.meas_screens[index]
#meas_screen.cutoff2(tracker.screen_cutoff)
#meas_screen.crop()
#meas_screen.reshape(tracker.len_screen)
#
#recon_kwargs['gaps'] = [10e-3, 10e-3+delta_gap]
#recon_kwargs['beam_offsets'] = [0., rec_obj_fb.beam_offsets[index]]
#recon_kwargs['n_streaker'] = 1
#recon_kwargs['meas_screen'] = meas_screen
#
#outp = analysis.current_profile_rec_gauss(tracker, recon_kwargs, plot_handles, blmeas_file)

if not args.noshow:
    ms.show()

if args.save:
    ms.saveall(args.save, hspace, wspace, ending='.pdf')

