import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

import gaussfit

class DoublehornFit:
    def __init__(self, xx, yy, raise_=True):

        self.xx = xx
        self.yy = yy

        gf = gaussfit.GaussFit(xx, yy, fit_const=False)

        arg_left = np.argmax(gf.yy[gf.xx < gf.mean])
        pos_left = gf.xx[arg_left]
        max_left = gf.yy[arg_left]


        mask_right = gf.xx > gf.mean
        arg_right = np.argmax(gf.yy[mask_right]) + np.sum(mask_right == 0)
        pos_right = gf.xx[arg_right]
        max_right = gf.yy[arg_right]
        s1a = s1b = s2a = s2b = gf.sigma / 5

        mask_middle = np.logical_and(gf.xx > pos_left, gf.xx < pos_right)
        if np.any(mask_middle):
            const_middle = np.min(gf.yy[mask_middle])
        else:
            const_middle = np.mean(yy)

        p0 = self.p0 = [pos_left, pos_right, s1a, s1b, s2a, s2b, const_middle, max_left, max_right]


        try:
            self.popt, self.pcov = curve_fit(self.fit_func, xx, yy, p0=p0)
        except RuntimeError as e:
            if raise_:
                plt.figure()
                plt.plot(xx, yy)
                plt.plot(xx, self.fit_func(xx, *p0))
                plt.plot(gf.xx, gf.reconstruction)
                plt.show()
                import pdb; pdb.set_trace()
            self.popt, self.pcov = p0, np.ones([len(p0), len(p0)], float)
            print(e)
            print('Fit did not converge. Using p0 instead!')

        self.reconstruction = self.fit_func(xx, *self.popt)
        #import pdb; pdb.set_trace()
        self.pos_left, self.pos_right, self.s1a, self.s1b, self.s2a, self.s2b, self.const_middle, self.max_left, self.max_right = self.popt

    @staticmethod
    def fit_func(xx, pos_left, pos_right, s1a, s1b, s2a, s2b, const_middle, max_left, max_right):
        outp = np.zeros_like(xx)
        pos_middle = (pos_right + pos_left)/2.

        mask1a = xx <= pos_left
        outp[mask1a] = gauss(xx[mask1a], max_left, pos_left, s1a, 0)

        mask1b = np.logical_and(xx > pos_left, xx <= pos_middle)
        outp[mask1b] = gauss(xx[mask1b], max_left-const_middle, pos_left, s1b, const_middle)

        mask2a = np.logical_and(xx > pos_middle, xx <= pos_right)
        outp[mask2a] = gauss(xx[mask2a], max_right-const_middle, pos_right, s2a, const_middle)

        mask2b = xx > pos_right
        outp[mask2b] = gauss(xx[mask2b], max_right, pos_right, s2b, 0)

        return outp


    def plot_data_and_fit(self, sp):
        sp.plot(self.xx, self.yy, label='Data', marker='.')
        sp.plot(self.xx, self.reconstruction, label='Reconstruction', marker='.', ls='--')
        sp.legend()

def gauss(xx, scale, mean, sig, const=0):
    #return scale*stats.norm.pdf(xx, mean, sig)
    if sig != 0:
        return scale*np.exp(-(xx-mean)**2/(2*sig**2))+const
    else:
        return 0

