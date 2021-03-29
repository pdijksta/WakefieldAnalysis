class Profile:

    def __init__(self):
        self._gf = None
        self._gf_xx = None
        self._gf_yy = None

    def compare(self, other):
        xx_min = min(self._xx.min(), other._xx.min())
        xx_max = max(self._xx.max(), other._xx.max())
        xx = np.linspace(xx_min, xx_max, max(len(self._xx), len(other._xx)))
        yy1 = np.interp(xx, self._xx, self._yy, left=0, right=0)
        yy2 = np.interp(xx, other._xx, other._yy, left=0, right=0)
        yy1 = yy1 / np.nanmean(yy1)
        yy2 = yy2 / np.nanmean(yy2)

        # modify least squares else large values dominate
        weight = yy1 + yy2
        np.clip(weight, weight.max()/2., None, out=weight)

        diff = ((yy1-yy2)/weight)**2

        outp = np.nanmean(diff[np.nonzero(weight)])
        #import pdb; pdb.set_trace()
        return outp

    def reshape(self, new_shape):
        _xx = np.linspace(self._xx.min(), self._xx.max(), int(new_shape))
        old_sum = np.sum(self._yy)

        _yy = np.interp(_xx, self._xx, self._yy)
        #if new_shape >= len(self._xx):
        #    _yy = np.interp(_xx, self._xx, self._yy)
        #else:
        #    _yy = np.histogram(self._xx, bins=int(new_shape), weights=self._yy)[0]
        #import pdb; pdb.set_trace()

        _yy *= old_sum / _yy.sum()
        self._xx, self._yy = _xx, _yy

    def cutoff(self, cutoff_factor):
        """
        Cutoff based on max value of the y array.
        """
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        abs_yy = np.abs(yy)
        yy[abs_yy<abs_yy.max()*cutoff_factor] = 0
        self._yy = yy / np.sum(yy) * old_sum

    def cutoff2(self, cutoff_factor):
        """
        Cutoff based on max value of the y array.
        Also sets to 0 all values before and after the first 0 value (from the perspective of the maximum).
        """
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        abs_yy = np.abs(yy)
        yy[abs_yy<abs_yy.max()*cutoff_factor] = 0

        index_max = np.argmax(abs_yy)
        index_arr = np.arange(len(yy))
        is0 = (yy == 0)
        zero_pos = np.logical_and(index_arr > index_max, is0)
        nearest_zero_pos = index_arr[zero_pos][0]
        zero_neg = np.logical_and(index_arr < index_max, is0)
        nearest_zero_neg = index_arr[zero_neg][-1]

        yy[:nearest_zero_neg] = 0
        yy[nearest_zero_pos:] = 0
        self._yy = yy / np.sum(yy) * old_sum


    def crop(self):
        old_sum = np.sum(self._yy)
        mask = self._yy != 0
        xx_nonzero = self._xx[mask]
        new_x = np.linspace(xx_nonzero.min(), xx_nonzero.max(), len(self._xx))
        new_y = np.interp(new_x, self._xx, self._yy)
        self._xx = new_x
        self._yy = new_y / new_y.sum() * old_sum

    def __len__(self):
        return len(self._xx)

    @property
    def gaussfit(self):
        if self._gf is not None and (self._xx.min(), self._xx.max(), self._xx.sum()) == self._gf_xx and (self._yy.min(), self._yy.max(), self._yy.sum()) == self._gf_yy:
            return self._gf

        self._gf = GaussFit(self._xx, self._yy, fit_const=False)
        self._gf_xx = (self._xx.min(), self._xx.max(), self._xx.sum())
        self._gf_yy = (self._yy.min(), self._yy.max(), self._yy.sum())
        return self._gf

    @property
    def doublehornfit(self):
        return doublehornfit.DoublehornFit(self._xx, self._yy)


    @property
    def integral(self):
        return np.trapz(self._yy, self._xx)

    def smoothen(self, gauss_sigma, extend=True):
        diff = self._xx[1] - self._xx[0]
        if extend:
            n_extend = int(gauss_sigma // diff * 2)
            extend0 = np.arange(self._xx[0]-(n_extend+1)*diff, self._xx[0]-0.5*diff, diff)
            extend1 = np.arange(self._xx[-1]+diff, self._xx[-1]+diff*(n_extend+0.5), diff)
            zeros0 = np.zeros_like(extend0)
            zeros1 = np.zeros_like(extend1)

            self._xx = np.concatenate([extend0, self._xx, extend1])
            self._yy = np.concatenate([zeros0, self._yy, zeros1])


        if gauss_sigma is None or gauss_sigma == 0:
            return
        real_sigma = gauss_sigma/diff
        new_yy = gaussian_filter1d(self._yy, real_sigma)
        self._yy = new_yy

        #if gauss_sigma < 5e-15:
        #    import pdb; pdb.set_trace()

    def center(self):
        self._xx = self._xx - self.gaussfit.mean

    def scale_xx(self, scale_factor, keep_range=False):
        new_xx = self._xx * scale_factor
        if keep_range:
            old_sum = self._yy.sum()
            new_yy = np.interp(self._xx, new_xx, self._yy, left=0, right=0)
            self._yy = new_yy / new_yy.sum() * old_sum
        else:
            self._xx = new_xx

    def scale_yy(self, scale_factor):
        self._yy = self._yy * scale_factor

    def remove0(self):
        mask = self._yy != 0
        self._xx = self._xx[mask]
        self._yy = self._yy[mask]

    def flipx(self):
        self._xx = -self._xx[::-1]
        self._yy = self._yy[::-1]

class ScreenDistribution(Profile):
    def __init__(self, x, intensity, real_x=None, subtract_min=True):
        super().__init__()
        self._xx = x
        assert np.all(np.diff(self._xx)>=0)
        self._yy = intensity
        if subtract_min:
            self._yy = self._yy - np.min(self._yy)
        self.real_x = real_x

    @property
    def x(self):
        return self._xx

    @property
    def intensity(self):
        return self._yy

    def normalize(self, norm=1.):
        self._yy = self._yy / self.integral * norm

    def plot_standard(self, sp, **kwargs):
        if self._yy[0] != 0:
            diff = self._xx[1] - self._xx[0]
            x = np.concatenate([[self._xx[0] - diff], self._xx])
            y = np.concatenate([[0.], self._yy])
        else:
            x, y = self.x, self.intensity

        if y[-1] != 0:
            diff = self.x[1] - self.x[0]
            x = np.concatenate([x, [x[-1] + diff]])
            y = np.concatenate([y, [0.]])

        return sp.plot(x*1e3, y/self.integral/1e3, **kwargs)

def getScreenDistributionFromPoints(x_points, screen_bins, smoothen=0):
    """
    Smoothening by applying changes to coordinate.
    Does not actually smoothen the output, just broadens it.
    """
    if smoothen:
        rand = np.random.randn(len(x_points))
        rand[rand>3] = 3
        rand[rand<-3] = -3
        x_points2 = x_points + rand*smoothen
    else:
        x_points2 = x_points
    screen_hist, bin_edges0 = np.histogram(x_points2, bins=screen_bins, density=True)
    screen_xx = (bin_edges0[1:] + bin_edges0[:-1])/2

    return ScreenDistribution(screen_xx, screen_hist, real_x=x_points)


