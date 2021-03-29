import numpy as np

import tracking
import analysis
import elegant_matrix
import misc

import myplotstyle as ms

ms.plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

basedir = '/mnt/data/data_2021-03-16/'
archiver_dir = '/mnt/data/archiver_api_data/'

streaker_calib_file = basedir+'2021_03_16-20_07_45_Calibration_SARUN18-UDCP020.h5'
streaker_calib = analysis.analyze_streaker_calibration(streaker_calib_file, False)
streaker_offset = streaker_calib['meta_data']['streaker_offset']

screen_calib_file = basedir+'2021_03_16-20_14_10_Screen_Calibration_data_SARBD02-DSCR050.h5'
screen_calib = analysis.analyze_screen_calibration(screen_calib_file, False)
x0 = screen_calib['x0']
beamsize = screen_calib['beamsize']

blmeas = basedir+'113875557_bunch_length_meas.h5'
blmeas = basedir+'113876237_bunch_length_meas.h5'

tt_halfrange = 200e-15
charge = 200e-12
screen_cutoff = 2e-3
profile_cutoff = 2e-2
len_profile = int(2e3)
struct_lengths = [1., 1.]
screen_bins = 400
smoothen = 30e-6
n_emittances = [900e-9, 500e-9]
n_particles = int(100e3)
n_streaker = 1
self_consistent = True
quad_wake = False
bp_smoothen = 1e-15
invert_offset = True
magnet_file = archiver_dir+'2021-03-16.h5'
timestamp = elegant_matrix.get_timestamp(2021, 3, 16, 20, 14, 10)
sig_t_range = np.arange(20, 50.01, 5)*1e-15
n_streaker = 1


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake, bp_smoothen=bp_smoothen)

profile_from_blmeas = tracking.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, subtract_min=True)
profile_from_blmeas.cutoff(3e-2)
profile_from_blmeas.crop()
profile_from_blmeas.reshape(len_profile)

emit_fit = tracker.fit_emittance(beamsize, smoothen, tt_halfrange)
print(emit_fit)
tracker.n_emittances = [emit_fit, emit_fit]

ms.figure('Forward propagation')
subplot = ms.subplot_factory(2, 2)
sp_ctr = 1

sp_pos = subplot(sp_ctr, title='Forward propagated', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr += 1
sp_neg = subplot(sp_ctr, title='Forward propagated', xlabel='x [mm]', ylabel='Intensity (arb. units)')
sp_ctr += 1

sp_blmeas1 = subplot(sp_ctr, title='Bunch length measurement neg', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

sp_blmeas2 = subplot(sp_ctr, title='Bunch length measurement pos', xlabel='t [fs]', ylabel='Current (arb. units)')
sp_ctr += 1

for sp_ in (sp_blmeas1, sp_blmeas2):
    profile_from_blmeas.plot_standard(sp_, color='black')


ny, nx = 2, 2
subplot = ms.subplot_factory(ny, nx)
sp_ctr = np.inf


offsets = streaker_calib['meta_data']['offsets']
x_axis = streaker_calib['raw_data']['pyscan_result']['x_axis']
all_reconstructions = []
true_offsets = []

lims_neg = -0.5, 3
lims_pos = -2, 0.5
sp_pos.set_xlim(*lims_pos)
sp_neg.set_xlim(*lims_neg)

delta_offset = 0e-6

center_plot = 'Right'

all_xt = []

for n_offset, offset in enumerate(offsets[:-1]):
    images = streaker_calib['raw_data']['pyscan_result']['image'][n_offset].astype(float)
    projections = images.sum(axis=-2)
    median_proj = misc.get_median(projections, method='gf_sigma')
    screen = misc.proj_to_screen(median_proj, x_axis, subtract_min=True)
    screen._xx = screen._xx - x0
    screen.cutoff(5e-2)
    screen.crop()
    screen.reshape(len_profile)

    beam_offsets = [0, -(offset - streaker_offset) + delta_offset]
    gaps = [10e-3, 10e-3]
    distance  = (gaps[1]/2. - abs(beam_offsets[1]))*np.sign(beam_offsets[1])

    forward_dict = tracker.matrix_forward(profile_from_blmeas, gaps, beam_offsets)
    screen_forward = forward_dict['screen']
    if offset == 0:
        continue
    elif offset > 0:
        sp = sp_pos
        sp_blmeas = sp_blmeas1
    else:
        sp = sp_neg
        sp_blmeas = sp_blmeas2

    label = '%i um' % (distance*1e6)
    color = screen.plot_standard(sp, label=label)[0].get_color()
    screen_forward.plot_standard(sp, color=color, ls='--')

    recon_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, screen, gaps, beam_offsets, n_streaker, charge)
    recon_profile = recon_dict['reconstructed_profile']
    recon_screen = recon_dict['reconstructed_screen']
    gap, beam_offset, struct_length = gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker]
    r12 = tracker.calcR12()[n_streaker]
    all_xt.append(recon_profile.get_x_t(gap, beam_offset, struct_length, r12))
    true_offsets.append(beam_offsets[1])

    recon_profile.plot_standard(sp_blmeas, label=label, center=center_plot)

    if sp_ctr > ny*nx:
        ms.figure('Details')
        sp_ctr = 1
    sp_screen = subplot(sp_ctr, title='%i um' % (distance*1e6), xlabel='x [mm]', ylabel='Intensity (arb. units)')
    sp_ctr += 1
    sp_profile = subplot(sp_ctr, title='%i um' % (distance*1e6), xlabel='t [fs]', ylabel='Current [kA]')
    sp_ctr += 1

    if offset > 0:
        sp_screen.set_xlim(*lims_pos)
    else:
        sp_screen.set_xlim(*lims_neg)

    screen.plot_standard(sp_screen, label='Meas')
    screen_forward.plot_standard(sp_screen, label='TDC forward')
    recon_screen.plot_standard(sp_screen, label='Reconstructed')

    recon_profile.plot_standard(sp_profile, label=label, center=center_plot)
    profile_from_blmeas.plot_standard(sp_profile, color='black', center=center_plot)

    sp_profile.legend()
    sp_screen.legend()

sp_blmeas1.legend()
sp_blmeas2.legend()

sp_neg.legend()
sp_pos.legend()

for profile, offset in zip(all_reconstructions, true_offsets):
    print('distance = %i um, beam duration = %i fs' % ((gaps[1]/2. - abs(offset))*1e6*np.sign(offset), profile.gaussfit.sigma*1e15))

#save_dict = {
#        'true_offsets': true_offsets,
#        'streaker_offsets': offsets,
#        'xt': all_xt,
#        'x0': x0,
#        }
#
#filename = './xt_2021-03-16.h5'
#from h5_storage import saveH5Recursive
#saveH5Recursive(filename, save_dict)
#print('Saved %s' % filename)



ms.plt.show()

