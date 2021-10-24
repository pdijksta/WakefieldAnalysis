#import sys
import numpy as np
import argparse
from socket import gethostname

#import streaker_calibration
import h5_storage
import elegant_matrix
import lasing
import config
import gaussfit
import image_and_profile as iap
#import tracking
#import analysis
import myplotstyle as ms

parser = argparse.ArgumentParser()
parser.add_argument('--noshow', action='store_true')
parser.add_argument('--save', type=str)
args = parser.parse_args()

config.fontsize=9
ms.set_fontsizes(config.fontsize)

np.random.seed(0)

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

main_fig = ms.figure('Main lasing', figsize=(13, 7.68))
hspace, wspace = 0.35, 0.3
ms.plt.subplots_adjust(hspace=hspace, wspace=wspace)
subplot = ms.subplot_factory(3, 3, grid=False)
sp_ctr = 1

rec_ctr = 2


norm_factor = None
for ctr, (lasing_on_file, lasing_off_file, pulse_energy, screen_x0, streaker_offset, curr_lim, main_title) in enumerate([
        (lasing_on_fileFB, lasing_off_fileFB, 625e-6, screen_x02, streaker_offset2, 1.3e3, 'Standard mode'),
        (lasing_on_file2, lasing_off_file2, 180e-6, screen_x02, streaker_offset2, 1.3e3, 'Double pulse'),
        (lasing_on_fileSB, lasing_off_fileSB, 85e-6, screen_x02, streaker_offset2, 1.3e3, 'Short pulse'),
        ]):

    lasing_off_dict = h5_storage.loadH5Recursive(lasing_off_file)
    lasing_on_dict = h5_storage.loadH5Recursive(lasing_on_file)


    n_streaker = 1
    beamline = 'Aramis'
    delta_gap = -63e-6
    tracker_kwargs = config.get_default_tracker_settings()
    recon_kwargs = config.get_default_gauss_recon_settings()
    slice_factor = 3
    charge = 180e-12
    subtract_median = False
    n_shots = 5
    recon_kwargs['charge'] = charge

    streaker = config.streaker_names['Aramis'][n_streaker]
    print('Streaker offset on/off: %.3f / %.3f mm ' % (lasing_on_dict['meta_data_begin'][streaker+':CENTER'], lasing_off_dict['meta_data_begin'][streaker+':CENTER']))

    las_rec_images = {}

    for main_ctr, (data_dict, title) in enumerate([(lasing_off_dict, 'Lasing Off'), (lasing_on_dict, 'Lasing On')]):
        rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=subtract_median, slice_factor=slice_factor)
        #if ctr == rec_ctr:
        #    rec_obj.do_recon_plot = True

        rec_obj.add_dict(data_dict)
        if main_ctr == 1:
            rec_obj.profile = las_rec_images['Lasing Off'].profile
            rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
            rec_obj.ref_y = np.mean(las_rec_images['Lasing Off'].ref_y_list)
        rec_obj.process_data()
        avg_distance = rec_obj.gap/2. - abs(rec_obj.beam_offsets.mean())
        print('Average distances (um)', (avg_distance*1e6))
        las_rec_images[title] = rec_obj
        #rec_obj.plot_images('raw', title)
        #rec_obj.plot_images('tE', title)

    las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=curr_lim, norm_factor=norm_factor)
    las_rec.plot(n_shots=n_shots)


    if ctr == 0:
        norm_factor = las_rec.norm_factor
    if ctr == rec_ctr:
        lasing_off_dict_fb = lasing_off_dict
        las_rec_fb = las_rec
        rec_obj_fb = las_rec_images['Lasing Off']

    ms.plt.figure(main_fig.number)
    title = main_title + ' ($E$=%i $\mu$J, $d$=%i $\mu$m)' % (round(las_rec.lasing_dict['Espread']['energy']*1e6), round(avg_distance*1e6))
    sp_off = subplot(ctr+1, title=title, xlabel='t (fs)', ylabel='E (MeV)', grid=False)
    sp_on = subplot(ctr+4, title=None, xlabel='t (fs)', ylabel='E (MeV)', grid=False)

    sp_espread = subplot(ctr+7, xlabel='t (fs)', ylabel='P (GW)')
    sp_dummy = lasing.dummy_plot()

    plot_handles = (sp_dummy, sp_dummy, sp_dummy, sp_dummy, sp_dummy, sp_dummy, sp_dummy, sp_espread, sp_dummy)
    las_rec.plot(plot_handles=plot_handles, n_shots=n_shots)
    #sp_espread.get_legend().remove()
    if ctr == 0:
        espread_ylim = [-2, sp_espread.get_ylim()[1]]
    sp_espread.set_ylim(*espread_ylim)

    for key, sp in [('Lasing On', sp_on), ('Lasing Off', sp_off)]:
        rec_obj = las_rec_images[key]
        image_tE = rec_obj.images_tE[0]

        rec_obj.slice_factor = 6
        rec_obj.slice_x()
        rec_obj.fit_slice()
        slice_dict = rec_obj.slice_dicts[0]
        image_tE.plot_img_and_proj(sp, plot_gauss=False, ylim=[-8e6, 6e6], slice_dict=slice_dict, slice_cutoff=curr_lim)


    if ctr == 0:
        gf_lims = []
        xx = las_rec.lasing_dict['Espread']['time']
        yy = las_rec.lasing_dict['Espread']['power']
        power_profile = iap.AnyProfile(xx, yy)
        print('Rms duration %.1f fs' % (power_profile.rms()*1e15))
        print('FWHM duration %.1f fs' % (power_profile.fwhm()*1e15))
    elif ctr == 1:
        gf_lims = ([30e-15, 45e-15], [60e-15, 80e-15])
    elif ctr == 2:
        gf_lims = ([40e-15, 55e-15],)

    for gf_ctr, gf_lim in enumerate(gf_lims):
        if gf_ctr == 0:
            ms.figure('Gf %s' % main_title)
            gf_subplot = ms.subplot_factory(2,2)
            gf_sp_ctr = 1
        sp = subplot(gf_sp_ctr, xlabel='t (fs)', ylabel='P (GW)')
        gf_sp_ctr += 1

        xx = las_rec.lasing_dict['Espread']['time']
        yy = las_rec.lasing_dict['Espread']['power']
        mask = np.logical_and(xx > gf_lim[0], xx < gf_lim[1])
        gf = gaussfit.GaussFit(xx[mask], yy[mask])
        gf.plot_data_and_fit(sp)
        sp.set_title('Gaussfit %i $\sigma$ %.1f fs m %.1f fs' % (gf_ctr, (gf.sigma*1e15), gf.mean*1e15))
        print('Gaussian duration %.1f fs' % (gf.sigma*1e15))
        print('Gaussian mean %.1f fs' % (gf.mean*1e15))
        #sp_espread.plot(gf.xx*1e15, gf.reconstruction/1e9, label='%.1f' % (gf.sigma*1e15), color='red', lw=3)

    #if gf_lims:
    #    sp_espread.legend(title='$\sigma$ (fs)')



if not args.noshow:
    ms.show()

if args.save:
    ms.saveall(args.save, hspace, wspace, ending='.pdf')

