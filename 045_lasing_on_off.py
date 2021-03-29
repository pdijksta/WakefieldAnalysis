import numpy as np

from h5_storage import loadH5Recursive
import myplotstyle as ms
import image_analysis
import data_loader
import elegant_matrix
import tracking
import misc

ms.plt.close('all')

data_dir = '/mnt/data/data_2021-03-16/'
archiver_dir = '/mnt/data/archiver_api_data/'
lasing_on_file = data_dir+'20210316_202944_SARBD02-DSCR050_camera_snapshot.h5'
lasing_off_file = data_dir+'20210316_204139_SARBD02-DSCR050_camera_snapshot.h5'
cutoff = 0.03
cutoff_proj = 0.1
x0 = 0.0005552048387736093

xt_file = './xt_2021-03-16.h5'
xt_dict = loadH5Recursive(xt_file)

streaker_data_file = archiver_dir+'2021-03-16_1.h5'
streaker_data = data_loader.DataLoader(file_h5=streaker_data_file)
streaker = 'SARUN18-UDCP020'


timestamp1 = elegant_matrix.get_timestamp(2021, 3, 16, 20, 29, 44)
timestamp2 = elegant_matrix.get_timestamp(2021, 3, 16, 20, 41, 39)

for timestamp in timestamp1, timestamp2:
    streaker_gap = streaker_data.get_prev_datapoint(streaker+':GAP', timestamp)*1e-3
    streaker_center = streaker_data.get_prev_datapoint(streaker+':CENTER', timestamp)*1e-3
    print('Streaker properties [mm]', timestamp, streaker_gap*1e3, streaker_center*1e3)


lasing_on = loadH5Recursive(lasing_on_file)
lasing_off = loadH5Recursive(lasing_off_file)

image_on = lasing_on['camera1']['image'].astype(float)
image_off = lasing_off['camera1']['image'].astype(float)

fig = ms.figure('See effect of lasing')
fig.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2, 2, grid=True)
sp_ctr = 1

x_axis = lasing_on['camera1']['x_axis'].astype(float)*1e-6 - x0
y_axis = lasing_on['camera1']['y_axis'].astype(float)*1e-6

if x_axis[1] < x_axis[0]:
    x_axis = x_axis[::-1]
    image_on = image_on[:,::-1]
    image_off = image_off[:,::-1]

if y_axis[1] < y_axis[0]:
    y_axis = y_axis[::-1]
    image_on = image_on[::-1,:]
    image_off = image_off[::-1,:]

sp_projx = subplot(sp_ctr, title='X projections', xlabel='x [mm]')
sp_ctr += 1
sp_projy = subplot(sp_ctr, title='Y projections', xlabel='y [mm]')
sp_ctr += 1

for image, label in [(image_off, 'Off'), (image_on, 'On')]:
    print(label, image.sum())
    if label == 'Off':
        med = np.median(image)
    image -= med
    sp_img = subplot(sp_ctr, title='Image %s' % label, xlabel='x [mm]', ylabel='y [mm]', grid=False)
    sp_ctr += 1

    extent = np.array([x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]])*1e3
    sp_img.imshow(image, aspect='auto', extent=extent, origin='lower')

    sp_projx.plot(x_axis*1e3, image.sum(axis=0), label=label)
    sp_projy.plot(y_axis*1e3, image.sum(axis=1), label=label)


ms.figure('Analyzer')
sp_ctr = 1

sp_max_slice = subplot(sp_ctr, title='Max slice')
sp_ctr += 1

sp_current1 = sp_max_slice.twinx()
sp_current1.set_ylabel('Current (arb. units)')
ms.sciy()

sp_gauss_std = subplot(sp_ctr, title='Gauss sigma')
sp_ctr += 1

sp_current2 = sp_gauss_std.twinx()
sp_current2.set_ylabel('Current (arb. units)')
ms.sciy()

for n_image, (image, label) in enumerate([(image_off, 'Off'), (image_on, 'On')]):

    analyzer = image_analysis.ImageAnalyzer(image, 10, x_axis, y_axis)
    projx = analyzer.image_averaged.sum(axis=0)

    if label == 'Off':
        mask = projx > cutoff_proj * projx.max()

    max_slice = analyzer.max_slice()
    gauss_slice, gauss_std = analyzer.gauss_slice(debug=False)

    x_axis_plot = analyzer.averaged_x_axis[mask]*1e3

    for sp_ in sp_current1, sp_current2:
        color = ['red', 'black'][n_image]
        sp_.plot(x_axis_plot, projx[mask], label=label+' Current', color=color)


    sp_max_slice.plot(x_axis_plot, max_slice[mask]*1e3, label=label)
    sp_max_slice.errorbar(x_axis_plot, gauss_slice[mask]*1e3, yerr=gauss_std[mask]*1e3, ls='--', label=label+' Gauss')
    sp_gauss_std.plot(x_axis_plot, gauss_std[mask]*1e3, label=label)

ms.comb_legend(sp_max_slice, sp_current1)
ms.comb_legend(sp_gauss_std, sp_current2)

sp_projx.legend()
sp_projy.legend()

## Compare images
#basedir = '/mnt/data/data_2021-03-16/'
#streaker_calib_file = basedir+'2021_03_16-20_07_45_Calibration_SARUN18-UDCP020.h5'
#streaker_calib = loadH5Recursive(streaker_calib_file)
#image_calib = streaker_calib['raw_data']['pyscan_result']['image'][-1][5].astype(float)
#image_calib = image_calib[::-1,::-1]
#
#ms.figure('Compare images')
#subplot = ms.subplot_factory(2,2, grid=False)
#sp_ctr = 1
#
#sp_img_nonlasing = subplot(sp_ctr, title='Lasing off')
#sp_ctr += 1
#extent = np.array([x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]])*1e3
#sp_img_nonlasing.imshow(image_off, aspect='auto', extent=extent, origin='lower')
#
#sp_img_calib = subplot(sp_ctr, title='From calib')
#sp_ctr += 1
#sp_img_calib.imshow(image_calib, aspect='auto', extent=extent, origin='lower')

# Conclusion: streaker calib and lasing off are a bit too different

# Get Current distribution

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
compensate_negative_screen = True


tracker = tracking.Tracker(magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_profile, quad_wake=quad_wake, bp_smoothen=bp_smoothen, compensate_negative_screen=compensate_negative_screen)

streaker_offset = 0.00037758839957521145

meas_screen = misc.image_to_screen(image_off, x_axis, True, x_offset=0)
meas_screen.cutoff2(5e-2)
meas_screen.crop()
meas_screen.reshape(len_profile)

ms.figure('Obtain x-t')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_screen = subplot(sp_ctr, title='Screen', xlabel='x axis', ylabel='Intensity (arb. units)')
sp_ctr += 1


sp_profile = subplot(sp_ctr, title='Profile', xlabel='t [fs]', ylabel='Current [kA]')
sp_ctr += 1

sp_tx = subplot(sp_ctr, title='t-x', xlabel='t[fs]', ylabel='x [mm]')
sp_ctr += 1


meas_screen.plot_standard(sp_screen)

gaps = [streaker_gap, streaker_gap]
beam_offsets = [0, -(streaker_center - streaker_offset)]

gauss_dict = tracker.find_best_gauss(sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge)

final_profile = gauss_dict['final_profile']
final_screen = gauss_dict['final_screen']

final_screen.plot_standard(sp_screen)
final_profile.plot_standard(sp_profile, center='Max')
final_profile._xx = final_profile._xx - final_profile._xx.min()

r12 = tracker.calcR12()[n_streaker]

tt, xx = final_profile.get_x_t(gaps[n_streaker], beam_offsets[n_streaker], struct_lengths[n_streaker], r12)

t_axis = np.interp(x_axis, xx, tt)

sp_tx.plot((tt-tt.min())*1e15, xx*1e3)

ms.figure('Backtracked images')
subplot = ms.subplot_factory(2,2, grid=False)
sp_ctr = 1

for image, label in [(image_on, 'Lasing on'), (image_off, 'Lasing off')]:

    x_mask = np.logical_and(x_axis >= xx.min(), x_axis <= xx.max())
    image_cut = image[:,x_mask]
    x_axis_cut = x_axis[x_mask]
    image2 = np.zeros([len(y_axis), len_profile])
    x_axis2 = np.linspace(x_axis_cut.min(), x_axis_cut.max(), len_profile)
    delta_x = np.zeros_like(image_cut)
    delta_x[:,:-1] = image_cut[:,1:] - image_cut[:,:-1]
    grid_points, points = x_axis_cut, x_axis2
    index_float = (points - grid_points[0]) / (grid_points[1] - grid_points[0])
    index = index_float.astype(int)
    index_delta = index_float-index
    np.clip(index, 0, len(grid_points)-1, out=index)
    image2 = image_cut[:, index] + index_delta * delta_x[:,index]


    sp = subplot(sp_ctr, title=label, xlabel='x [mm]', ylabel='y [mm]')
    sp_ctr += 1
    extent = np.array([xx.min(), xx.max(), y_axis[0], y_axis[-1]])*1e3
    sp.imshow(np.log(image2), aspect='auto', extent=extent, origin='lower')
    proj = image2.sum(axis=-2)
    #proj_interp = np.interp(xx, x_axis[x_mask], proj)
    proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*1e3
    sp.plot(x_axis2*1e3, proj_plot, color='red')

    new_img = np.zeros_like(image2)
    new_t_axis = np.linspace(t_axis.min(), t_axis.max(), new_img.shape[1])
    diff_t = np.concatenate([[0], np.diff(tt)])
    diff_interp = np.interp(new_t_axis, tt, diff_t)
    x_interp = np.interp(new_t_axis, tt, xx)
    all_x_index = np.zeros_like(new_t_axis, dtype=int)
    for t_index, (t, x, diff) in enumerate(zip(new_t_axis, x_interp, diff_interp)):
        x_index = np.argmin((x_axis2 - x)**2)
        all_x_index[t_index] = x_index
        new_img[:,t_index] = 0 if diff == 0 else image2[:,x_index]/diff

    new_img = new_img[:,::-1]
    new_img = new_img/new_img.sum()*image2.sum()



    sp = subplot(sp_ctr, title=label, xlabel='t [fs]', ylabel='y [mm]')
    sp_ctr += 1
    extent = [new_t_axis.max()*1e15, new_t_axis.min()*1e15, y_axis[0]*1e3, y_axis[-1]*1e3]
    sp.imshow(np.log(new_img), aspect='auto', extent=extent, origin='lower')

    proj = new_img.sum(axis=-2)
    proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*1e3
    sp.plot(new_t_axis[::-1]*1e15, proj_plot, color='red')
    current = final_profile.current
    time = final_profile.time - final_profile.time.min()
    curr_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * current/current.max()*0.3)*1e3
    sp.plot((time*1e15)[::-1], curr_plot, color='orange')


#import pickle
#with open('./backtrack_image_no_compensate.pkl', 'wb') as f:
#    pickle.dump({
#        'image': image_off,
#        'x_axis': x_axis,
#        'y_axis': y_axis,
#        'final_profile': final_profile,
#        'tt': tt,
#        'xx': xx,
#        }, f)

ms.plt.show()

