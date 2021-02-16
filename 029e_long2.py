import os
#import mat73
import copy; copy
import socket
import numpy as np; np
import matplotlib.pyplot as plt
#from scipy.constants import c

import elegant_matrix
import tracking
import gaussfit
import misc
from h5_storage import loadH5Recursive

import myplotstyle as ms

plt.close('all')

hostname = socket.gethostname()
elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

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
quad_wake = True
bp_smoothen = 1e-15
#sig_t_range = np.arange(20, 40.01, 2)*1e-15

#mean_struct2 = 472e-6 # see 026_script
fudge_factor = 0
mean_struct2 = 466e-6 + fudge_factor
gap2_correcting_summand = 0 #-3e-6
sig_t_range = np.arange(20, 40.01, 5)*1e-15
gaps = [10e-3, 10e-3]
subtract_min = True
fit_emittance = True

# According to Alex, use data from these days:
# https://elog-gfa.psi.ch/SwissFEL+commissioning/16450 (4th October 2020)
# https://elog-gfa.psi.ch/SwissFEL+commissioning/16442 (3rd October 2020)

# 3rd October
if hostname == 'desktop':
    dirname1 = '/storage/data_2020-10-03/'
    dirname2 = '/storage/data_2020-10-04/'
    archiver_dir = '/storage/Philipp_data_folder/'
elif hostname == 'pc11292.psi.ch':
    dirname1 = '/sf/data/measurements/2020/10/03/'
    dirname2 = '/sf/data/measurements/2020/10/04/'
elif hostname == 'pubuntu':
    dirname1 = '/home/work/data_2020-10-03/'
    dirname2 = '/home/work/data_2020-10-04/'
    archiver_dir = '/home/work/'

magnet_file = archiver_dir + 'archiver_api_data/2020-10-03.h5'

blmeas38 = dirname1+'129833611_bunch_length_meas.h5'
blmeas25 = dirname2 + '129918532_bunch_length_meas.h5'
blmeas25 = dirname2 + 'Bunch_length_meas_2020-10-04_14-43-30.h5'
blmeas13 = dirname2 + '129920069_bunch_length_meas.h5'


file0 = dirname1 + 'Passive_data_20201003T231958.mat'

file38 = dirname1 + 'Passive_data_20201003T233852.mat'
file25 = dirname2 + 'Passive_data_20201004T172425.mat'
file13 = dirname2 + 'Passive_data_20201004T221502.mat'

dict38 = loadH5Recursive(file38+'.h5')
dict25 = loadH5Recursive(file25+'.h5')
dict0 = loadH5Recursive(file0+'.h5')
dict13_h5 = loadH5Recursive(file13+'.h5')
dict13_h5['value'] = np.array(-4.25) # from archiver
#dict13_h5['x_axis'] = dict25['x_axis']

proj13 = dict13_h5['projx']
proj13_new = np.reshape(proj13, (proj13.shape[0]*proj13.shape[1], proj13.shape[2]))
dict13_h5['projx'] = proj13_new

def get_screen_from_proj(projX, x_axis, invert_x):
    if invert_x:
        xx, yy = (-x_axis[::-1]).copy(), (projX[::-1]).copy()
    else:
        xx, yy = x_axis.copy(), projX.copy()
    if subtract_min:
        yy -= yy.min()
    screen = tracking.ScreenDistribution(xx, yy)
    screen.normalize()
    screen.cutoff(screen_cutoff)
    screen.reshape(len_profile)
    return screen


process_dict = {
        'Long': {
            'filename': file0,
            'main_dict': dict0,
            'proj0': dict0['projx'][-1],
            'x_axis0': dict0['x_axis']*1e-6,
            'n_offset': 0,
            'filename0': file0,
            'blmeas': blmeas38,
            'flipx': False,
            'limits_screen': [-0.5e-3, 1.5e-3],
            'center': 'Right_fit',
        },
        'Medium': {
            'filename': file25,
            'main_dict': dict25,
            'proj0': dict25['projx'][7],
            'x_axis0': dict25['x_axis']*1e-6,
            'n_offset': 0,
            'filename0': file25,
            'blmeas': blmeas25,
            'flipx': False,
            'limits_screen': [-0.5e-3, 2.5e-3],
            'center': 'Gauss',
            },
        'Short': {
            'filename': file13,
            'main_dict': dict13_h5,
            'proj0': '0',
            'x_axis0': 0,
            'n_offset': None,
            'filename0': None,
            'blmeas': blmeas13,
            'flipx': False,
            'limits_screen': [-0.5e-3, 2.5e-3],
            'center': 'Gauss',
            },

        }

for main_label, p_dict in process_dict.items():
    if main_label == 'Short':
        continue

    #fig_paper = ms.figure('Old %s' % main_label)
    #subplot = ms.subplot_factory(2, 2)
    #sp_ctr_paper = 1

    if type(p_dict['proj0']) is str:
        mean0 = -0.08e-3
    else:
        projx0 = p_dict['proj0']
        x_axis0 = p_dict['x_axis0']
        if np.diff(x_axis0)[0] < 0:
            x_axis0 = x_axis0[::-1]
            invert_x0 = True

        all_mean = []
        for proj in projx0:
            screen = get_screen_from_proj(proj, x_axis0, invert_x0)
            xx, yy = screen._xx, screen._yy
            gf = gaussfit.GaussFit(xx, yy)
            all_mean.append(gf.mean)

        mean0 = np.mean(all_mean)
        print('%s: Mean0: %.3e mm' % (main_label, mean0*1e3))

    if fit_emittance and main_label != 'Short':

        timestamp0 = misc.get_timestamp(os.path.basename(p_dict['filename0']))
        tracker0 = tracking.Tracker(magnet_file, timestamp0, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake)

        bp_test = tracking.get_gaussian_profile(40e-15, tt_halfrange, len_profile, charge, tracker0.energy_eV)
        screen_sim = tracker0.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']
        all_emittances = []
        all_beamsizes = []
        for proj in projx0:
            screen_meas = get_screen_from_proj(proj, x_axis0, invert_x0)
            all_beamsizes.append(screen_meas.gaussfit.sigma)
            emittance_fit = misc.fit_nat_beamsize(screen_meas, screen_sim, n_emittances[0], smoothen, print_=False)
            #print(screen_meas.gaussfit.sigma)
            all_emittances.append(emittance_fit)

        new_emittance = np.mean(all_emittances)
        print(main_label, 'Emittance [nm]', new_emittance*1e9)

        n_emittances[0] = new_emittance

        tracker0.n_emittances[0] = new_emittance

        new_screen0 = tracker0.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen']
        ms.figure('Test nat bs')
        sp = plt.subplot(1,1,1)
        sp.set_title('New emittance %i nm' % (new_emittance*1e9))
        screen_meas.center()
        for screen, label in [(new_screen0, 'New'), (screen_meas, 'Meas'), (screen_sim, 'Initial')]:
            color = screen.plot_standard(sp, label=label)[0].get_color()
            xx, yy = screen.gaussfit.xx, screen.gaussfit.reconstruction
            sp.plot(xx*1e3, yy/np.trapz(yy, xx), color=color, ls='--', label='%i' % (screen.gaussfit.sigma*1e6))
        sp.legend()
    else:
        new_emittance = n_emittances[0]

    fig_paper = ms.figure('For paper %s ($\epsilon$=%i nm)' % (main_label, new_emittance*1e9))
    fig_paper.subplots_adjust(hspace=0.3)
    subplot = ms.subplot_factory(2, 3)
    sp_ctr_p = 1

    sp_tdc_meas = subplot(sp_ctr_p, title='Current profiles', xlabel='time [fs]', ylabel='Current (A)')
    sp_ctr_p += 1

    sp_backtrack_tdc_screen = subplot(sp_ctr_p, title='TDC forward', xlabel='x [mm]', ylabel='Screen projection')
    sp_ctr_p += 1

    sp_back_forward = subplot(sp_ctr_p, title='Reconstruction forward', xlabel='x [mm]', ylabel='Screen projection')
    sp_ctr_p += 1

    sp_recon = subplot(sp_ctr_p, title='Optimized current profiles', xlabel='time [fs]', ylabel='Current (A)')
    sp_ctr_p += 1

    sp_recon_screen = subplot(sp_ctr_p, title='Optimized projections', xlabel='x [mm]', ylabel='Screen projection')
    sp_ctr_p += 1


    dict_ = p_dict['main_dict']
    file_ = p_dict['filename']
    x_axis = dict_['x_axis']*1e-6
    n_offset = p_dict['n_offset']
    center = p_dict['center']

    if np.diff(x_axis)[0] < 0:
        x_axis = x_axis[::-1]
        invert_x = True
    else:
        invert_x = False

    timestamp  = misc.get_timestamp(os.path.basename(file_))
    tracker = tracking.Tracker(archiver_dir + 'archiver_api_data/2020-10-03.h5', timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake, bp_smoothen=bp_smoothen)

    blmeas = p_dict['blmeas']
    flip_measured = p_dict['flipx']
    profile_meas = tracking.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, subtract_min=True)
    profile_meas.reshape(len_profile)
    profile_meas2 = tracking.profile_from_blmeas(blmeas, tt_halfrange, charge, tracker.energy_eV, subtract_min=True, zero_crossing=2)
    profile_meas2.reshape(len_profile)
    if flip_measured:
        profile_meas.flipx()
    else:
        profile_meas2.flipx()

    profile_meas.cutoff(2e-2)
    profile_meas2.cutoff(2e-2)
    profile_meas.crop()
    profile_meas2.crop()

    beam_offsets = [0., -(dict_['value']*1e-3 - mean_struct2)]
    distance_um = (gaps[n_streaker]/2. - beam_offsets[n_streaker])*1e6
    if n_offset is not None:
        distance_um = distance_um[n_offset]
        beam_offsets = [beam_offsets[0], beam_offsets[1][n_offset]]

    tdc_screen1 = tracker.matrix_forward(profile_meas, gaps, beam_offsets)['screen']
    tdc_screen2 = tracker.matrix_forward(profile_meas2, gaps, beam_offsets)['screen']

    #plt.figure(fig_paper.number)
    #sp_profile_comp = subplot(sp_ctr_paper, title=main_label, xlabel='t [fs]', ylabel='Intensity (arb. units)')
    #sp_ctr_paper += 1
    #profile_meas.plot_standard(sp_profile_comp, norm=True, color='black', label='TDC', center=center)


    ny, nx = 2, 4
    subplot = ms.subplot_factory(ny, nx)
    sp_ctr = np.inf

    all_profiles, all_screens = [], []

    if n_offset is None:
        projections = dict_['projx']
    else:
        projections = dict_['projx'][n_offset]

    ## FINAL PLOTS

    for sp_ in sp_tdc_meas, sp_recon:
        profile_meas.plot_standard(sp_, center=center, label='TDC', color='black')
    #profile_meas2.plot_standard(sp_tdc_meas, center='Left_fit', label='TDC 2')
    sig_list = []
    for n_image, projx in enumerate(projections):
        screen = get_screen_from_proj(projx, x_axis, invert_x)
        sig_list.append(screen.gaussfit.sigma)

    index_avg = np.argsort(sig_list)[len(sig_list)//2]
    median_screen = get_screen_from_proj(projections[index_avg], x_axis, invert_x)
    median_screen._xx = median_screen._xx - mean0
    median_screen.cutoff(1e-2)
    limits = p_dict['limits_screen']
    mask_median = np.logical_or(median_screen.x > limits[1], median_screen.x < limits[0])
    median_screen._yy[mask_median] = 0
    median_screen.crop()
    for sp_ in sp_backtrack_tdc_screen, sp_recon_screen, sp_back_forward:
        median_screen.plot_standard(sp_, label='Median', color='black')

    median_screen_sigma = median_screen.gaussfit.sigma

    all_screens = {'median_screen': median_screen}

    for q_wake, label in [(False, 'Dipole'), (True, '+Quadrupole')]:

        tracker.quad_wake = q_wake
        profile_tdc_back = tracker.track_backward2(median_screen, profile_meas, gaps, beam_offsets, n_streaker, plot_details=True)
        profile_tdc_back.plot_standard(sp_tdc_meas, center=center, label=label)
        screen_forward1 = tracker.matrix_forward(profile_meas, gaps, beam_offsets)['screen_no_smoothen']
        screen_forward1.smoothen(smoothen)
        screen_forward1.cutoff(screen_cutoff)
        color = screen_forward1.plot_standard(sp_backtrack_tdc_screen, label=label)[0].get_color()
        print('Difference for %s TDC: %e' % (label, median_screen.compare(screen_forward1)))
        all_screens['%s_TDC' % label] = screen_forward1

        screen_forward2 = tracker.matrix_forward(profile_tdc_back, gaps, beam_offsets)['screen_no_smoothen']
        screen_forward2.smoothen(smoothen)
        screen_forward2.cutoff(screen_cutoff)
        screen_forward2.plot_standard(sp_back_forward, color=color, label=label)

        gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, median_screen, gaps, beam_offsets, n_streaker, charge)
        bp_recon = gauss_dict['reconstructed_profile']

        #scale_dict = tracker.scale_existing_profile(

        screen_recon = gauss_dict['reconstructed_screen_no_smoothen']
        screen_recon.smoothen(smoothen)
        screen_recon.cutoff(screen_cutoff)

        bp_recon.plot_standard(sp_recon, label=label, center=center)
        #bp_recon.plot_standard(sp_tdc_meas, color=color, center=center, ls='--')
        screen_recon.plot_standard(sp_recon_screen, label=label)
        print('Difference for %s recon: %e' % (label, median_screen.compare(screen_recon)))
        all_screens['%s_recon' % label] = screen_recon


    sp_backtrack_tdc_screen.legend()
    sp_tdc_meas.legend()
    sp_recon.legend()
    sp_recon_screen.legend()
    sp_back_forward.legend()

ms.saveall('/tmp/plot_for_paper', hspace=0.3, wspace=0.3)

plt.show()

