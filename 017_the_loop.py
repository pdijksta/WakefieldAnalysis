import socket
import itertools
import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt

from h5_storage import loadH5Recursive
from EmittanceTool.gaussfit import GaussFit
import wf_model
import elegant_matrix

import myplotstyle as ms

plt.close('all')

# Optimizer around elegant tracking

# 1.
#   - Get initial distribution by fitting 4 parameters:
#   - heights of left spike, right spike, middle
#   - width of spikes
#   - Assume known chamber offset

# 2.
#   - Fit chamber offset as well (using images from different offsets with known delta)

# 3.
#   - Fit 21 slice heights
#

# 4.
# Better approach
# Strategy:
# 1. Wakefield for any simulated or measured profiles can be well approximated by 5-th order polynomial function (≤ 5 parameters)
# 2. Guess the initial wakefield model (single-gaussian wake, etc.)
# 3. Back-propagate the measured profile to the time domain
# 4. Calculate the real wakefield using model
# 5. Forward track to the screen (Model-based code or Elegant)
# 6. Compare with measurements
# 7. Adjust the wakefield model
# 8. Loop steps 2-7 = optimization problem



# files from alex

files_labels = [
        ('SpectrumAnalysis_2020_07_26_17_56_25_401050.h5', 'no streaking'),
        ('SpectrumAnalysis_2020_07_26_17_55_28_676729.h5', '-4.04 mm dump'),
        ('SpectrumAnalysis_2020_07_26_17_46_52_221317.h5', '-4.09 mm dump'),
        ]
energy_eV = 6.14e9
gap = 10e-3
fit_order = 4
show_4_images = False
zoom_sig = 3

elegant_matrix.set_tmp_dir('/home/philipp/tmp_elegant/')

hostname = socket.gethostname()
if hostname == 'desktop':
    save_directory = '/storage/data_2020-07-26/'
    other = '/storage/data_2020-10-04/20201004_175405_undulator_wf.h5'
    magnet_file = '/storage/Philipp_data_folder/archiver_api_data/2020-07-26.h5'
else:
    save_directory = '/sf/data/measurements/2020/07/26/'
    other = '/sf/data/measurements/2020/10/04/20201004_175405_undulator_wf.h5'
    magnet_file = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-07-26.h5'

simulator = elegant_matrix.get_simulator(magnet_file)
timestamp = elegant_matrix.get_timestamp(2020, 7, 26, 17, 49, 0)
mat_dict, disp_dict = simulator.get_elegant_matrix(0, timestamp)
r12 = mat_dict['SARBD02.DSCR050'][0,1]

other_dict = loadH5Recursive(other)
x_axis = other_dict['x_axis'].astype(np.float64) * 1e-6
y_axis = other_dict['y_axis'].astype(np.float64) * 1e-6
del other, other_dict


### Generate Gaussian beam to calculate initial wake potential

# approximate 40 fs beam
sig_t = 40e-15
time = np.linspace(0., 400e-15, 1000)

curr = np.exp(-(time-np.mean(time))**2/(2*sig_t**2))
curr[curr<0.001*curr.max()] = 0

curr = curr * 200e-12/np.sum(curr)
wf_calc = wf_model.WakeFieldCalculator(time*c, curr, energy_eV, 1.)
wf_dict = wf_calc.calc_all(gap/2., r12, beam_offset=(-4.09e-3-0.47e-3), calc_lin_dipole=True, calc_dipole=False, calc_quadrupole=False, calc_long_dipole=False)

wake_zz = wf_dict['input']['charge_xx']
convoluted_wake = wf_dict['lin_dipole']['wake_potential'] * -1

ms.figure('Wakefields')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_wake = subplot(sp_ctr, title='Initial wake from Gaussian $\sigma_t$=%i fs' % (sig_t*1e15), xlabel='s [$\mu$m]', ylabel='kV/m')
sp_ctr += 1

sp_wake.plot(wake_zz*1e6, convoluted_wake/1e3, label='Wake')

poly_fit = np.poly1d(np.polyfit(wake_zz, convoluted_wake, fit_order))
fitted_wake = poly_fit(wake_zz)
sp_wake.plot(wake_zz*1e6, fitted_wake/1e3, label='Fit')

sp_wake.legend()

def zoom_image(image, x_axis, y_axis, n_sig):
    mask = {}
    gf_dict = {}
    for axis, proj, label in [(x_axis, np.sum(image, 0), 'X'), (y_axis, np.sum(image, 1), 'Y')]:
        gf = GaussFit(axis, proj)
        min_ax, max_ax = gf.mean - gf.sigma * n_sig, gf.mean + gf.sigma * n_sig
        mask[label] = np.logical_and(axis > min_ax, axis < max_ax)
        gf_dict[label] = gf

    return {
            'image': image[mask['Y']][:,mask['X']],
            'x_axis': x_axis[mask['X']],
            'y_axis': y_axis[mask['Y']],
            'gf_dict': gf_dict,
            }

if show_4_images:

    for file_, file_label in files_labels:

        meas_dict = loadH5Recursive(save_directory+file_)

        images = meas_dict['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'].astype(np.float64)
        ms.figure('Check image %s' % file_label)

        subplot = ms.subplot_factory(2,2)
        sp_ctr = 1

        for n_image in range(4):

            image = images[n_image,0]
            image_dict = zoom_image(image, x_axis, y_axis, zoom_sig)

            image_zoom = image_dict['image']
            ax_x = image_dict['x_axis']
            ax_y = image_dict['y_axis']
            gf_dict = image_dict['gf_dict']


            sp = subplot(sp_ctr, title='Image sig=%i $\mu$m' % (gf_dict['X'].sigma*1e6), grid=False)
            sp_ctr += 1
            sp.imshow(image_zoom, aspect='auto', extent=(ax_x[0], ax_x[-1], ax_y[-1], ax_y[0]))

file_ = files_labels[-1][0]
meas_dict = loadH5Recursive(save_directory+file_)
images = meas_dict['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'].astype(np.float64)

ms.figure('All projX')
subplot = ms.subplot_factory(2,3)
sp_ctr = 1
sp = subplot(sp_ctr, title='All projections', xlabel='Screen X [$\mu$m]', ylabel='Intensity (arb. units)')
sp_ctr += 1

proj_list, axis_list, mean_X_list = [], [], []

for n1, n2 in itertools.product(range(images.shape[0]), range(images.shape[1])):
    image = images[n1, n2]
    zoom_dict = zoom_image(image, x_axis, y_axis, zoom_sig)
    image2 = zoom_dict['image']
    x_axis2 = zoom_dict['x_axis'] # - zoom_dict['gf_dict']['X'].mean
    projX = np.sum(image2, axis=0)
    proj_list.append(projX)
    axis_list.append(x_axis2)
    mean_X_list.append(zoom_dict['gf_dict']['X'].mean)

    sp.plot(x_axis2*1e6, projX)

mean_X = np.mean(mean_X_list)

file0 = files_labels[0][0]
meas_dict0 = loadH5Recursive(save_directory+file0)
images0 = meas_dict0['scan_1']['data']['SARBD02-DSCR050']['FPICTURE'].astype(np.float64)

mean_X0_list = []
for n1, n2 in itertools.product(range(images0.shape[0]), range(images0.shape[1])):
    image = images0[n1, n2]
    gf = GaussFit(x_axis, np.sum(image, axis=0))
    mean_X0_list.append(gf.mean)

mean_X0 = np.mean(mean_X0_list)

min_x = min(np.min(axis) for axis in axis_list)
max_x = max(np.max(axis) for axis in axis_list)

comb_axis = np.linspace(min_x, max_x, int(200))
interp_proj = np.zeros([len(proj_list), len(comb_axis)])
for n_axis, (axis, proj) in enumerate(zip(axis_list, proj_list)):
    interp_proj[n_axis] = np.interp(comb_axis, axis[::-1], proj[::-1])

comb_proj = np.mean(interp_proj, axis=0)
comb_proj_std = np.std(interp_proj, axis=0)

x_screen_axis = comb_axis - mean_X0
x_screen = comb_proj

xp_struct_axis = -x_screen_axis / r12
xp_struct = comb_proj
xp_struct_std = comb_proj_std

sp = subplot(sp_ctr, title='Average X Screen', xlabel='Screen X [$\mu$m]', ylabel='Intensity (arb. units)')
sp_ctr += 1
sp.errorbar(comb_axis*1e6, comb_proj, yerr=comb_proj_std)
sp.axvline(mean_X0*1e6, color='red')
sp.axvline(mean_X*1e6, color='blue')

sp = subplot(sp_ctr, title='Average Xp structure', xlabel='Structure Xp [mrad]', ylabel='Intensity (arb. units)')
sp_ctr += 1
sp.errorbar(xp_struct_axis*1e3, xp_struct, yerr=xp_struct_std)

wf_normalized = convoluted_wake / energy_eV

sp = subplot(sp_ctr, title='Normalized wake', xlabel='z [$\mu$m]', ylabel='Xp [mrad]')
sp_ctr += 1
sp.plot(wake_zz*1e6, wf_normalized*1e3)

interp_zz = np.interp(-xp_struct_axis, wf_normalized, wake_zz)

sp = subplot(sp_ctr, title='Interpolated Z', xlabel='z [$\mu$m]', ylabel='Intensity (arb. units)')
sp_ctr += 1
sp.plot(interp_zz*1e6, xp_struct)

plt.show()

