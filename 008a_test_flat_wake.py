import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
#import numpy.fft as fft

import uwf_model as uwf
import myplotstyle as ms

plt.close('all')

pi = np.pi
sigma = uwf.sigma_cu
ctau = uwf.ctau_cu


#integral_re = uwf.impedance_flat_re(1, 3e-3)

a = 2.5e-3
s0 = uwf.s0(a, uwf.sigma_cu)
potency_2 = 13

n_s0 = 10
n_samples = int(1e3)

fig = ms.figure('Details of flat wake calc')
subplot = ms.subplot_factory(2, 3)
sp_ctr = 1


for s0_factor in range(0, 5):
    s0_factor *= 2
    k = s0_factor/s0
    sigma_tilde = sigma / (1 - 1j*k*ctau)
    lamb = np.sqrt((2*pi*sigma*np.abs(k))/c) * (1j + np.sign(k))
    lamb_tilde = np.sqrt((2*pi*sigma_tilde*np.abs(k))/c) * (1j + np.sign(k))

    sp = subplot(sp_ctr, title='k*s0=%i' % s0_factor)
    sp_ctr += 1

    xx = np.linspace(-10000, 10000, n_samples)
    yy_re = np.array(list(map(lambda x: uwf.impedance_flat_integrand(x, a, lamb, -1, uwf.RE), xx)))
    yy_im = np.array(list(map(lambda x: uwf.impedance_flat_integrand(x, a, lamb, -1, uwf.IM), xx)))


    yy_re_tilde = np.array(list(map(lambda x: uwf.impedance_flat_integrand(x, a, lamb_tilde, -1, uwf.RE), xx)))
    yy_im_tilde = np.array(list(map(lambda x: uwf.impedance_flat_integrand(x, a, lamb_tilde, -1, uwf.IM), xx)))


    trapz_re = np.trapz(yy_re, xx)
    int_re, err_re = uwf.quad(uwf.impedance_flat_integrand, xx.min(), xx.max(), (a, lamb, -1, uwf.RE))
    #print('%e' % trapz_re, '%e' % int_re)

    sp.plot(xx, yy_re, label='RE dc')
    sp.plot(xx, yy_im, label='IM dc')
    sp.plot(xx, yy_re_tilde, label='RE ac')
    sp.plot(xx, yy_im_tilde, label='IM ac')

    sp.legend()


#fig = ms.figure('Wake integrand')
#sp_ctr = 1
#
#for ss in np.arange(0, 50.01, 10)*1e-6:
#
#    sp = subplot(sp_ctr, title='Wake integrand s=%i $\mu$m' % (ss*1e6), scix=True)
#    sp_ctr += 1
#
#    xx = np.linspace(0, 20/s0, n_samples)
#    integrand_wake_flat = np.array(list(map(lambda x: uwf.integrand_wake_flat_cos_dc(x, a, ss), xx)))
#
#    sp.plot(xx, integrand_wake_flat)
#


fig = ms.figure('Impedance / Wake flat ac')
sp_ctr = 1

k_factor = 10

k_arr = np.linspace(-k_factor/s0, k_factor/s0, 2**potency_2)
re_arr = np.zeros_like(k_arr)
im_arr = re_arr.copy()
re_err_arr = re_arr.copy()
im_err_arr = re_arr.copy()
re_dc_arr = re_arr.copy()
im_dc_arr = re_arr.copy()


for n_k, k in enumerate(k_arr):
    re, im, err_re, err_im = uwf.impedance_flat_ac(k, a)
    re_arr[n_k] = re
    im_arr[n_k] = im
    re_err_arr[n_k] = err_re
    im_err_arr[n_k] = err_im

    re_dc_arr[n_k], im_dc_arr[n_k] = uwf.impedance_flat_re_dc(k, a)

sp = subplot(sp_ctr, title='Impedance Gap %.1f mm' % (a*1e3))
sp_ctr += 1

x_factor = s0
y_factor = c*a**2/s0
sp.errorbar(k_arr*x_factor, re_arr*y_factor, yerr=re_err_arr*y_factor, label='AC RE')
sp.errorbar(k_arr*x_factor, im_arr*y_factor, yerr=im_err_arr*y_factor, label='AC IM')
sp.plot(k_arr*x_factor, re_dc_arr*y_factor, label='DC RE')
sp.plot(k_arr*x_factor, im_dc_arr*y_factor, label='DC IM')
sp.legend()

sp = subplot(sp_ctr, title='Wake AC', sciy=False, xlabel='S [$\mu$m]')
sp_ctr += 1

#complex_impedance = re_arr + 1j*im_arr
# reorder...
#complex_impedance2 = np.zeros_like(complex_impedance)
#complex_impedance2[:len(k_arr)//2] = complex_impedance[len(k_arr)//2:]
#complex_impedance2[len(k_arr)//2:] = complex_impedance[:len(k_arr)//2]
#
#complex_impedance = complex_impedance2

#complex_impedance = fft.ifftshift(complex_impedance)


#wake = fft.ifft(complex_impedance) * c
#ss = -fft.fftfreq(len(wake), np.diff(k_arr)[0]/2/np.pi) # factor 1/2pi because k corresponds to omega, not f, while dft expects f...


#mask = np.logical_and(ss <= 150e-6, ss >= 0)
#mask[0] = False
x_factor = 1e6
y_factor = a**2/4



#sp.plot(ss[mask]*x_factor, wake.imag[mask]*y_factor, label='IM')


# Try Cosine transform
s_arr = np.linspace(0, 150e-6, int(1e3))
W_arr = np.zeros_like(s_arr)
W_arr2 = np.zeros_like(s_arr)
#W_arr3_re = np.zeros_like(s_arr)
#W_arr3_im = np.zeros_like(s_arr)
k_arr0 = np.linspace(0, k_factor/s0, 2**(potency_2-1))

W_arr_comp = uwf.wake_flat_ac(s_arr, a)

for n_s, s in enumerate(s_arr):
    cos_integrand = re_arr[2**(potency_2-1):] * np.cos(k_arr0*s)
    sin_integrand = im_arr[2**(potency_2-1):] * np.sin(k_arr0*s)
    #fourier_integrand = (re_arr + 1j*im_arr) * np.exp(-1j*k_arr*s)
    W_arr[n_s] = 2*c/np.pi * np.trapz(cos_integrand, k_arr0)
    W_arr2[n_s] = 2*c/np.pi * np.trapz(sin_integrand, k_arr0)
    #W_arr3_re[n_s] = c/(2*np.pi) * np.trapz(fourier_integrand.real.astype(float), k_arr)
    #W_arr3_im[n_s] = c/(2*np.pi) * np.trapz(fourier_integrand.imag.astype(float), k_arr)


sp.plot(s_arr*x_factor, W_arr*y_factor, label='Cos')
sp.plot(s_arr*x_factor, W_arr2*y_factor, label='Sin')
sp.plot(W_arr_comp['s_arr']*x_factor, W_arr_comp['W_arr']*y_factor, label='Cos 2')
#sp.plot(s_arr*x_factor, W_arr3_re*y_factor, label='RE')
#sp.plot(s_arr*x_factor, W_arr3_im*y_factor, label='IM')

#norm_factor = W_arr.max()/wake.real[mask].max()
#sp.plot(ss[mask]*x_factor, wake.real[mask]*y_factor*norm_factor, label='RE')


sp.legend()

plt.show()

