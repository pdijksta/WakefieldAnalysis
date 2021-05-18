import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from socket import gethostname

import h5_storage
import misc2 as misc
import analysis
from analysis import streaker_calibration_figure
import myplotstyle as ms

do_plot = True
plot_handles = None
fit_order = True



plt.close('all')

hostname = gethostname()
if hostname == 'desktop':
    data_dir = '/storage/data_2021-04-25/'
elif hostname == 'pc11292.psi.ch':
    data_dir = '/sf/data/measurements/2021/04/25/'
elif hostname == 'pubuntu':
    data_dir = '/storage/data_2021-04-25/'


calib_file = data_dir + '2021_04_25-16_55_25_Calibration_SARUN18-UDCP020.h5'
screen_file = data_dir + '2021_04_25-16_58_20_Screen_Calibration_SARBD02-DSCR050.h5'

offset_file_dict = {
        5.06: '/sf/data/measurements/2021/04/25/2021_04_25-17_16_30_Lasing_False_SARBD02-DSCR050.h5',
        5.04: '/sf/data/measurements/2021/04/25/2021_04_25-17_33_31_Lasing_False_SARBD02-DSCR050.h5',
        5.02: '/sf/data/measurements/2021/04/25/2021_04_25-17_36_28_Lasing_False_SARBD02-DSCR050.h5',
        5.00: '/sf/data/measurements/2021/04/25/2021_04_25-17_37_20_Lasing_False_SARBD02-DSCR050.h5',
        -4.4: '/sf/data/measurements/2021/04/25/2021_04_25-17_40_15_Lasing_False_SARBD02-DSCR050.h5',
        -4.42: '/sf/data/measurements/2021/04/25/2021_04_25-17_41_26_Lasing_False_SARBD02-DSCR050.h5',
        -4.38: '/sf/data/measurements/2021/04/25/2021_04_25-17_42_51_Lasing_False_SARBD02-DSCR050.h5',
        -4.36: '/sf/data/measurements/2021/04/25/2021_04_25-17_44_59_Lasing_False_SARBD02-DSCR050.h5',
        }


screen_analysis = analysis.analyze_screen_calibration(screen_file)
screen_center = screen_analysis['x0']
print('Screen center module', screen_center*1e6, 'um')

calib_dict = h5_storage.loadH5Recursive(calib_file)
raw_data = calib_dict['raw_data']

result0 = analysis.analyze_streaker_calibration(calib_dict, True)
print('Module streaker center', result0['meta_data']['streaker_offset']*1e6, 'um')

mask = np.logical_and(raw_data['streaker_offsets'] < 5.04e-3, raw_data['streaker_offsets'] > -0.0044)
#mask = np.ones_like(raw_data['streaker_offsets'], dtype=bool)
raw_data['streaker_offsets'] = raw_data['streaker_offsets'][mask]
raw_data['pyscan_result']['image'] = raw_data['pyscan_result']['image'][mask]
print(raw_data['streaker_offsets'])


ms.figure('Screen projections')
subplot = ms.subplot_factory(1,1)
sp_proj = subplot(1, title='Screen projections', xlabel='x (mm)', ylabel='Intensity (arb. units)')

screen_data = h5_storage.loadH5Recursive(screen_file)
image = screen_data['pyscan_result']['image'][0]
screen = misc.image_to_screen(image, screen_data['pyscan_result']['x_axis'], True, 0)

screen.plot_standard(sp_proj, color='black', label='Screen calib')
sp_proj.axvline(screen_analysis['x0']*1e3, color='black')


#for res in result0, result1:
#    print(int(res['meta_data']['streaker_offset']*1e6))

def streaker_calibration_fit_func(offsets, streaker_offset, strength, order=3, const=0, semigap=0):
    wall0, wall1 = -semigap, semigap
    c1 = np.abs((offsets-streaker_offset-wall0))**(-order)
    c2 = np.abs((offsets-streaker_offset-wall1))**(-order)
    return const + (c1 - c2)*strength




filename_or_dict = raw_data
if type(filename_or_dict) is dict:
    data_dict = filename_or_dict
elif type(filename_or_dict) is str:
    data_dict = h5_storage.loadH5Recursive(filename_or_dict)
else:
    raise ValueError(type(filename_or_dict))

if 'raw_data' in data_dict:
    data_dict = data_dict['raw_data']
result_dict = data_dict['pyscan_result']

images = result_dict['image'].astype(float).squeeze()
proj_x = np.sum(images, axis=-2).squeeze()
x_axis = result_dict['x_axis_m']
offsets = data_dict['streaker_offsets'].squeeze()
n_images = int(data_dict['n_images'])

centroids = np.zeros([len(offsets), n_images])
if n_images > 1:
    for n_o, n_i in itertools.product(range(len(offsets)), range(n_images)):
        this_proj = proj_x[n_o,n_i].copy()
        this_proj -= np.median(this_proj)
        this_proj[this_proj<0] = 0
        mask0 = np.abs(this_proj) < np.abs(this_proj).max()*0.05
        this_proj[mask0] = 0
        centroids[n_o,n_i] = np.sum(this_proj*x_axis) / np.sum(this_proj)
        if n_i == 0:
            screen = misc.proj_to_screen(this_proj, x_axis, True, 0)
            color = screen.plot_standard(sp_proj, label='%i' % (offsets[n_o]*1e3))[0].get_color()
            sp_proj.axvline(centroids[n_o,n_i]*1e3, color=color, ls='--')

    centroid_mean = np.median(centroids, axis=1)
    centroid_std = np.std(centroids, axis=1)
elif n_images == 1:
    for n_o in range(len(offsets)):
        centroids[n_o] = np.sum(proj_x[n_o]*x_axis) / np.sum(proj_x[n_o])
    centroid_mean = centroids.squeeze()
    centroid_std = None

streaker = data_dict['streaker']
semigap = data_dict['meta_data'][streaker+':GAP']/2.*1e-3

wall0, wall1 = -semigap, semigap

where0 = np.argwhere(offsets == 0).squeeze()
const0 = centroid_mean[where0]
delta_offset0 = (offsets.min() + offsets.max())/2
order0 = 3

s01 = (centroid_mean[0] - const0) / (np.abs((offsets[0]-delta_offset0-wall0))**(-order0) - np.abs((offsets[0]-delta_offset0-wall1))**(-order0))
s02 = (centroid_mean[-1] - const0) / (np.abs((offsets[-1]-delta_offset0-wall0))**(-order0) - np.abs((offsets[-1]-delta_offset0-wall1))**(-order0))
strength0 = s01



if fit_order:
    p0 = [delta_offset0, strength0, order0]
else:
    p0 = [delta_offset0, strength0]

def fit_func(*args):
    if fit_order:
        args2 = args[:-1]
        return streaker_calibration_fit_func(*args2, order=args[-1], semigap=semigap, const=const0)
    else:
        return streaker_calibration_fit_func(*args, semigap=semigap, const=const0)

try:
    p_opt, p_cov = curve_fit(fit_func, offsets, centroid_mean, p0, sigma=centroid_std)
except RuntimeError:
    print('Streaker calibration did not converge')
    p_opt = p0
xx_fit = np.linspace(offsets.min(), offsets.max(), int(1e3))
reconstruction = fit_func(xx_fit, *p_opt)
initial_guess = fit_func(xx_fit, *p0)
streaker_offset = p_opt[0]
screen_offset = const0


meta_data = {
        'p_opt': p_opt,
        'p0': p0,
        'centroid_mean': centroid_mean,
        'centroid_std': centroid_std,
        'offsets': offsets,
        'semigap': semigap,
        'streaker_offset': streaker_offset,
        'reconstruction': reconstruction,
        }

output = {
        'raw_data': data_dict,
        'meta_data': meta_data,
        }

if do_plot:

    if plot_handles is None:
        fig, (sp_center, sp_proj2) = streaker_calibration_figure()
    else:
        (sp_center, ) = plot_handles
    screen = data_dict['screen']
    sp_center.set_title(screen)

    xx_plot = (offsets - streaker_offset)*1e3
    xx_plot_fit = (xx_fit - streaker_offset)*1e3
    sp_center.errorbar(xx_plot, (centroid_mean-screen_offset)*1e3, yerr=centroid_std*1e3, label='Data', ls='None')
    sp_center.plot(xx_plot_fit, (reconstruction-screen_offset)*1e3, label='Fit')
    sp_center.plot(xx_plot_fit, (initial_guess-screen_offset)*1e3, label='Guess')


new_images = []
new_offsets = []
for offset_mm, file_ in sorted(offset_file_dict.items()):
    file_ = data_dir + os.path.basename(file_)
    dd = h5_storage.loadH5Recursive(file_)
    image = dd['pyscan_result']['image'][0]
    x_axis = dd['pyscan_result']['x_axis']
    screen = misc.image_to_screen(image, x_axis, True, screen_offset)
    screen.plot_standard(sp_proj2, label='%.2f' % offset_mm)
    new_images.append(dd['pyscan_result']['image'])
    new_offsets.append(offset_mm*1e-3)


new_dict = {
        'streaker_offsets': np.array(new_offsets),
        'screen': data_dict['screen'],
        'n_images': len(new_images[-1]),
        'streaker': data_dict['streaker'],
        'pyscan_result': {
            'x_axis_m': x_axis,
            'image': np.array(new_images),
            },
        'meta_data': data_dict['meta_data'],
        }




sp_proj2.set_title('From main scan')
sp_proj2.legend()
sp_center.legend()


print('Streaker center', meta_data['streaker_offset']*1e6)
print('Screen center', screen_offset*1e6)
if fit_order:
    print('Order', p_opt[-1])

sp_proj.legend()


new_analysis = analysis.analyze_streaker_calibration(new_dict, force_screen_center=const0)
print('New analysis streaker offset', new_analysis['meta_data']['streaker_offset']*1e6, 'um')

plt.show()

