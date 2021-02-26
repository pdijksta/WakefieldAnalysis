import pickle
from scipy.constants import c
import numpy as np; np
from scipy import interpolate
import wf_model

import myplotstyle as ms

ms.plt.close('all')


with open('./quad_interp.pkl', 'rb') as f:
    _, _, _, beam = pickle.load(f)

beam_offset = 0.0047

beam_x = beam[0,:]
beam_s = beam[4,:] * c


wake2d_dict_quad = wf_model.wf2d_quad(beam_s, beam_x+beam_offset, 5e-3, 200e-12, wf_model.wxq, hist_bins=(1000, 500))

quad_s = wake2d_dict_quad['s_bins']
quad_x = wake2d_dict_quad['x_bins']
quad_wake = wake2d_dict_quad['wake']




indices = []
for grid_points, points in [(quad_s, beam_s), (quad_x, beam_x+beam_offset)]:
    index_float = (points - grid_points[0]) / (grid_points[1] - grid_points[0])
    index = np.round(index_float).astype(int)
    np.clip(index, 0, len(grid_points)-1, out=index)
    indices.append(index)

vals = quad_wake[indices[0], indices[1]]

vals2 = wake2d_dict_quad['wake_on_particles']


interp_quad = interpolate.interp2d(quad_s, quad_x, quad_wake.T)

quad_effect = np.zeros_like(vals)
for n, (s_, x) in enumerate(zip(beam_s, beam_x+beam_offset)):
    quad_effect[n] = interp_quad(s_, x)



ms.figure('Test interp')
subplot = ms.subplot_factory(1, 2)

sp = subplot(1)

sp.hist((vals-quad_effect)/np.mean(np.abs(quad_effect)), bins=100)
sp.set_yscale('log')


sp = subplot(2)

sp.hist((vals2-quad_effect)/np.mean(np.abs(quad_effect)), bins=100)
sp.set_yscale('log')








ms.plt.show()

