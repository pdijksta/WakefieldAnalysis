import copy
import numpy as np
import pickle

import misc
import tracking

import myplotstyle as ms

ms.plt.close('all')

def plot_img_and_proj(sp, image, x_axis, y_axis, x_factor=1e3, y_factor=1e3, plot_proj=True, log=True):
    extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[0]*y_factor, y_axis[-1]*y_factor]
    if log:
        image_ = np.clip(image, 1, None)
        #image_ = image
        log = np.log(image_)
    else:
        log = image
    sp.imshow(log, aspect='auto', extent=extent, origin='lower')
    if plot_proj:
        proj = image.sum(axis=-2)
        proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*y_factor
        sp.plot(x_axis*x_factor, proj_plot, color='red')

with open('./backtrack_image.pkl', 'rb') as f:
    d = pickle.load(f)
    image = d['image']
    x_axis = d['x_axis']
    y_axis = d['y_axis']
    final_profile = d['final_profile']
    xx = d['xx']
    tt = d['tt']

t_axis = np.interp(x_axis, xx, tt)

len_profile = len(final_profile)

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

#image2 *= image_cut.sum()/image2.sum()

figure = ms.figure('Backtrack image')
ms.plt.subplots_adjust(hspace=0.3)
subplot = ms.subplot_factory(2,3)
sp_ctr = 1


sp = subplot(sp_ctr, title='X space 1', xlabel='x [mm]', ylabel='y [mm]')
sp_ctr += 1
plot_img_and_proj(sp, image_cut, x_axis_cut, y_axis)

sp = subplot(sp_ctr, title='X space 2', xlabel='x [mm]', ylabel='y [mm]')
sp_ctr += 1
plot_img_and_proj(sp, image2, x_axis2, y_axis)


new_img0 = np.zeros_like(image2)
new_t_axis = np.linspace(t_axis.min(), t_axis.max(), new_img0.shape[1])
diff_t = np.concatenate([[0], np.diff(tt)])
diff_interp = np.interp(new_t_axis, tt, diff_t)
x_interp = np.interp(new_t_axis, tt, xx)
all_x_index = np.zeros_like(new_t_axis, dtype=int)
for t_index, (t, x, diff) in enumerate(zip(new_t_axis, x_interp, diff_interp)):
    x_index = np.argmin((x_axis2 - x)**2)
    all_x_index[t_index] = x_index
    new_img0[:,t_index] = image2[:,x_index]

diff_x = np.concatenate([np.diff(x_interp), [0]])

new_img = new_img0 * diff_x

new_img = new_img / new_img.sum() * image2.sum()



new_img = new_img[:,::-1]
new_img = new_img/new_img.sum()*image2.sum()



sp = subplot(sp_ctr, title='T space', xlabel='t [fs]', ylabel='y [mm]')
sp_ctr += 1
plot_img_and_proj(sp, new_img, new_t_axis, y_axis, x_factor=1e15)

proj = final_profile.current
proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*1e3
sp.plot(new_t_axis*1e15, proj_plot, color='orange')


sp_screen = subplot(sp_ctr, title='Screen')
sp_ctr += 1

subtract_min = False
meas_screen = misc.image_to_screen(image, x_axis, subtract_min, x_offset=0)
meas_screen_cut = misc.image_to_screen(image_cut, x_axis_cut, subtract_min, x_offset=0)
meas_screen2 = misc.image_to_screen(image2, x_axis2, subtract_min, x_offset=0)

wake_x = xx
wake_time = tt

if wake_x[1] < wake_x[0]:
    wake_x = wake_x[::-1]
    wake_time = wake_time[::-1]

n_particles = int(100e3)
charge = 200e-12


sp_profile = subplot(sp_ctr, title='Profiles')
sp_ctr += 1

for image, axis, label in [(image, x_axis, 'Original'), (image_cut, x_axis_cut, 'Cut'), (image2, x_axis2, '2')]:

    screen = misc.image_to_screen(image, axis, subtract_min, x_offset=0)
    screen.plot_standard(sp_screen, label=label)
    screen = copy.deepcopy(screen)
    screen.reshape(n_particles)
    t_interp0 = np.interp(screen.x, wake_x, wake_time)
    charge_interp, hist_edges = np.histogram(t_interp0, bins=n_particles//100, weights=screen.intensity, density=True)
    charge_interp[0] = 0
    charge_interp[-1] = 0
    t_interp = np.linspace(t_interp0[0], t_interp0[-1], len(charge_interp))

    if t_interp[1] < t_interp[0]:
        t_interp = t_interp[::-1]
        charge_interp = charge_interp[::-1]
    t_interp -= t_interp.min()
    bp = tracking.BeamProfile(t_interp, charge_interp, 1, charge)
    bp.plot_standard(sp_profile, label=label)



final_profile.plot_standard(sp_profile, label='From module')

profile_arr = np.interp(new_t_axis, final_profile.time, final_profile.current)

new_img_sum = new_img.sum(axis=-2)
new_img_sum[new_img_sum == 0] = np.inf
image3 = new_img / new_img_sum / profile_arr.sum() * profile_arr
image3 = image3 / image3.sum() * new_img.sum()
sp = subplot(sp_ctr, 'From profile', xlabel='time [fs]')
sp_ctr += 1
plot_img_and_proj(sp, image3, new_t_axis, y_axis, x_factor=1e15)

sp_screen.legend()
sp_profile.legend()


ms.figure('Debug approach')
subplot = ms.subplot_factory(2,2)
sp_ctr = 1

sp_diff = subplot(sp_ctr)
sp_ctr += 1

sp_diff.plot(new_t_axis, diff_t)
sp_diff.plot(new_t_axis[:-1], np.diff(tt))


ms.plt.show()

