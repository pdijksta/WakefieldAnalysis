import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import deconvolve, convolve

import tracking

import myplotstyle as ms

plt.close('all')

with open('./screens.pkl', 'rb') as f:
    s_dict = pickle.load(f)

screen_meas = s_dict['screen_meas']
screen_meas0 = s_dict['screen_meas0']
screen_meas00 = s_dict['screen_meas00']
screen_meas000 = s_dict['screen_meas000']

for screen in screen_meas, screen_meas0, screen_meas00, screen_meas000:
    #screen.smoothen(60e-6)
    screen.reshape(10000)

screen_meas._xx -= screen_meas0.gaussfit.mean
screen_meas00._xx -= screen_meas000.gaussfit.mean

beamsize = screen_meas0.gaussfit.sigma

xx_arr = screen_meas00.x
gauss = np.exp(-(xx_arr-np.mean(xx_arr))**2/(2*beamsize**2))/(np.sqrt(2*np.pi)*beamsize)

mask_gauss = gauss>0.2*gauss.max()
cut_gauss = gauss[mask_gauss]
cut_xx = xx_arr[mask_gauss]
cut_gauss /= cut_gauss.sum()

convoluted_screen = convolve(screen_meas00.intensity, cut_gauss)
diff = np.diff(xx_arr).mean()
convolved_xx = np.arange(0, len(convoluted_screen)*diff, diff) - len(cut_gauss)/2*diff


zero_arr = np.zeros([len(cut_gauss)//2])
intensity_padded = np.concatenate([zero_arr, screen_meas.intensity, zero_arr[:-1]])
#deconvolved, remainder = deconvolve(intensity_padded, cut_gauss)


#random_distortion = np.random.randn(len(convoluted_screen))*0.01*convoluted_screen.max()
#random_distortion = 0
deconvolved, remainder = deconvolve(intensity_padded, cut_gauss)

screen_decon = tracking.ScreenDistribution(screen_meas.x, deconvolved)
screen_decon.smoothen(54e-6)







ms.figure('Screens')

subplot = ms.subplot_factory(2,2)
sp_ctr = 1


sp0 = subplot(1, title='Screens')

sp0.plot(screen_meas.x, screen_meas.intensity, label='Streak 300 nm')
sp0.plot(screen_meas0.x, screen_meas0.intensity/screen_meas0.intensity.max()*screen_meas.intensity.max(), label='No Streak 300 nm')
sp0.plot(screen_meas00.x, screen_meas00.intensity, label='Streak 1 nm')
sp0.plot(convolved_xx, convoluted_screen, label='1 nm streak x 300 nm no streak')
sp0.plot(screen_decon.x, screen_decon.intensity, label='Deconvoluted Streak 300 nm')

sp0.legend()


plt.show()










