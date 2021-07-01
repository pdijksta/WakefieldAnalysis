from socket import gethostname
import numpy as np

import elegant_matrix
import h5_storage
import lasing
import config
import myplotstyle as ms

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

n_streaker = 1
beamline = 'Aramis'
delta_gap = -62e-6
tracker_kwargs = config.get_default_tracker_settings()
recon_kwargs = config.get_default_gauss_recon_settings()
charge = 200e-12
streaker_offset = 364e-6
screen_x0 = 898.02e-6
pulse_energy = 180e-6
slice_factor = 5

lasing_file = data_dir1+'2021_05_18-21_41_35_Lasing_True_SARBD02-DSCR050.h5'
lasing_file2 = data_dir1+'2021_05_18-21_45_00_Lasing_False_SARBD02-DSCR050.h5'

lasing_dict_on = h5_storage.loadH5Recursive(lasing_file)
lasing_dict_off = h5_storage.loadH5Recursive(lasing_file2)

las_rec_images = {}
for main_ctr, (data_dict, title) in enumerate([(lasing_dict_off, 'Lasing Off'), (lasing_dict_on, 'Lasing On')]):
    rec_obj = lasing.LasingReconstructionImages(screen_x0, beamline, n_streaker, streaker_offset, delta_gap, tracker_kwargs, recon_kwargs=recon_kwargs, charge=charge, subtract_median=True, slice_factor=3)
    rec_obj.do_recon_plot = False

    rec_obj.add_dict(data_dict, max_index=None)
    if main_ctr == 0:
        rec_obj.process_data()
    if main_ctr == 1:
        rec_obj.profile = las_rec_images['Lasing Off'].profile
        rec_obj.ref_slice_dict = las_rec_images['Lasing Off'].ref_slice_dict
        rec_obj.process_data()

    las_rec_images[title] = rec_obj
    offset_dicts = rec_obj.get_streaker_offsets()
    #rec_obj.plot_images('raw', title)
    rec_obj.plot_images('tE', title)
    #figs, subplots = rec_obj.plot_images('slice', title)
    #subplots[0][0].axvline(rec_obj.wake_t.min()*1e15, color='red')
    #subplots[0][0].axvline(rec_obj.wake_t.max()*1e15, color='red')

las_rec = lasing.LasingReconstruction(las_rec_images['Lasing Off'], las_rec_images['Lasing On'], pulse_energy, current_cutoff=1.0e3)
las_rec.plot(plot_loss=True)

plot_delta_offset = False

if plot_delta_offset:
    final_sim_screens = []
    meas_screens = []
    delta_offsets = []

    for offset_dict in offset_dicts:
        final_sim_screens.append(offset_dict['sim_screen'])
        meas_screens.append(offset_dict['meas_screen'])
        delta_offsets.append(offset_dict['delta_offset'])

        ms.figure('Investigate offset dict')
        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1

        sp_screen = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel='Intensity (arb. units)')
        sp_ctr += 1

        sp_moments = subplot(sp_ctr, title='Screen moments', xlabel='$\Delta$d ($\mu$m)', ylabel=r'$\left|\langle x \rangle\right|$, $\sqrt{\langle x^2\rangle}$ (mm)')
        sp_ctr += 1


        offset_dict['meas_screen'].plot_standard(sp_screen, color='black', lw=3, label='Measured')
        for beam_offset, screen in zip(offset_dict['beam_offsets'], offset_dict['sim_screens']):
            screen.plot_standard(sp_screen, label='%.2f' % (beam_offset*1e3))

        delta_arr = offset_dict['beam_offsets']-offset_dict['beam_offset0']
        color = sp_moments.plot(delta_arr*1e3, offset_dict['rms_arr']*1e3, marker='.', label='rms')[0].get_color()
        sp_moments.axhline(offset_dict['meas_screen'].rms()*1e3, color=color, ls='--')
        color = sp_moments.plot(delta_arr*1e3, np.abs(offset_dict['mean_arr']*1e3), marker='.', label='mean')[0].get_color()
        sp_moments.axhline(np.abs(offset_dict['meas_screen'].mean())*1e3, color=color, ls='--')

    delta_offsets = np.array(delta_offsets)

    ms.figure('Delta offset overview')
    subplot = ms.subplot_factory(2,2)
    sp_ctr = 1

    sp_screen_overview = subplot(sp_ctr, title='Screen', xlabel='x (mm)', ylabel='Intensity (arb. units)')
    sp_ctr += 1

    mean_list = []

    for meas_screen, sim_screen, delta_offset in zip(meas_screens, final_sim_screens, delta_offsets):
        color = meas_screen.plot_standard(sp_screen_overview, label='%i $\mu$m' % (delta_offset*1e6))[0].get_color()
        sim_screen.plot_standard(sp_screen_overview, color=color, ls='--')
        mean_list.append(sim_screen.mean())
    mean_list = np.array(mean_list)

    sp_delta_overview = subplot(sp_ctr, title='Offset jitter', xlabel=r'$\left|\langle x \rangle\right|$', ylabel='$\Delta$d ($\mu$m)')
    sp_ctr += 1

    sort = np.argsort(mean_list)
    sp_delta_overview.plot(np.abs(mean_list[sort])*1e3, delta_offsets[sort]*1e6, marker='.')




ms.show()


