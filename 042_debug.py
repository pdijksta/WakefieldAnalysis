import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import analysis
import gaussfit
from h5_storage import loadH5Recursive

import myplotstyle as ms

plt.close('all')

filename = '/tmp/2021_03_16-20_22_26_Screen_data_SARBD02-DSCR050.h5'
dict_ = loadH5Recursive(filename)


ms.figure('Images')

subplot = ms.subplot_factory(3,3, grid=False)
sp_ctr = 1


sp_proj_x = subplot(sp_ctr, title='X')
sp_ctr += 1


sp_proj_y = subplot(sp_ctr, title='Y')
sp_ctr += 1

x_axis = dict_['pyscan_result']['x_axis'].astype(float)
y_axis = dict_['pyscan_result']['y_axis'].astype(float)


def fit_func(xx, scale, mean, sig, const):
    return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const

for n_img in range(4):

    image0 = dict_['pyscan_result']['image'][n_img].astype(float)

    if n_img == 0:
        projx = image0.sum(axis=-2)
        projx -= projx.min()
        sp_proj_x.plot(x_axis, projx)
        gf = gaussfit.GaussFit(x_axis, projx, fit_const=True)
        p0 = list(gf.p0)
        p0[2] *= 5
        p_opt, p_cov = curve_fit(fit_func, x_axis, projx, p0=p0)
        yy = fit_func(x_axis, *p_opt)
        sp_proj_x.plot(x_axis, yy)
        sp_proj_x.plot(gf.xx, gf.reconstruction)
        sp_proj_y.plot(y_axis, image0.sum(axis=-1))

    sp_img = subplot(sp_ctr)
    sp_ctr += 1
    sp_img.imshow(image0, aspect='auto')











plt.show()

