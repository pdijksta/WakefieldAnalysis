import numpy as np
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d

try:
    from . import data_loader
    from .gaussfit import GaussFit
    from . import doublehornfit
    from . import wf_model
    from . import misc2 as misc
    from . import myplotstyle as ms
except ImportError:
    import data_loader
    from gaussfit import GaussFit
    import doublehornfit
    import wf_model
    import misc2 as misc
    import myplotstyle as ms

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

class BeamProfile(Profile):
    def __init__(self, time, current, energy_eV, charge):
        super().__init__()

        if np.any(np.isnan(time)):
            raise ValueError('nans in time')
        if np.any(np.isnan(current)):
            raise ValueError('nans in current')

        self._xx = time
        assert np.all(np.diff(self._xx)>=0)
        self._yy = current / current.sum() * charge
        self.energy_eV = energy_eV
        self.charge = charge
        self.wake_dict = {}

    @property
    def time(self):
        return self._xx

    @property
    def current(self):
        return self._yy

    def scale_yy(self, scale_factor):
        self.charge *= scale_factor
        super().scale_yy(scale_factor)

    def calc_wake(self, gap, beam_offset, struct_length):

        if abs(beam_offset) > gap/2.:
            raise ValueError('Beam offset is too large!')

        if (gap, beam_offset, struct_length) in self.wake_dict:
            return self.wake_dict[(gap, beam_offset, struct_length)]

        wf_calc = wf_model.WakeFieldCalculator((self.time - self.time.min())*c, self.current, self.energy_eV, struct_length)
        wf_dict = wf_calc.calc_all(gap/2., R12=0., beam_offset=beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=True, calc_long_dipole=True)

        self.wake_dict[(gap, beam_offset, struct_length)] = wf_dict

        return wf_dict

    def get_x_t(self, gap, beam_offset, struct_length, r12):
        wake_dict = self.calc_wake(gap, beam_offset, struct_length)
        wake_effect = self.wake_effect_on_screen(wake_dict, r12)
        tt, xx = wake_effect['t'], wake_effect['x']
        tt = tt - tt.min()
        return tt, xx

    def write_sdds(self, filename, gap, beam_offset, struct_length):
        s0 = (self.time - self.time[0])*c
        new_s = np.linspace(0, s0.max()*1.1, int(len(s0)*1.1))
        w_wld = wf_model.wld(new_s, gap/2., beam_offset)*struct_length
        if beam_offset == 0:
            w_wxd = np.zeros_like(w_wld)
        else:
            w_wxd = wf_model.wxd(new_s, gap/2., beam_offset)*struct_length
        w_wxd_deriv = np.zeros_like(w_wxd)
        return wf_model.write_sdds(filename, new_s/c, w_wld, w_wxd, w_wxd_deriv)


    def wake_effect_on_screen(self, wf_dict, r12):
        wake = wf_dict['dipole']['wake_potential']
        quad = wf_dict['quadrupole']['wake_potential']
        wake_effect = wake/self.energy_eV*r12*np.sign(self.charge)
        quad_effect = quad/self.energy_eV*r12*np.sign(self.charge)
        output = {
                't': self.time,
                'x': wake_effect,
                'quad': quad_effect,
                'charge': self.charge,
                }
        return output

    def shift(self, center):
        if center == 'Max':
            center_index = np.argmax(np.abs(self.current))
        elif center == 'Left':
            center_index = misc.find_rising_flank(np.abs(self.current))
        elif center == 'Right':
            center_index = len(self.current) - misc.find_rising_flank(np.abs(self.current[::-1]))
        elif center == 'Right_fit':
            dhf = doublehornfit.DoublehornFit(self._xx, self._yy)
            center_index = np.argmin((self._xx - dhf.pos_right)**2)
        elif center == 'Left_fit':
            dhf = doublehornfit.DoublehornFit(self._xx, self._yy)
            center_index = np.argmin((self._xx - dhf.pos_left)**2)
        else:
            raise ValueError(center)

        self._xx = self._xx - self._xx[center_index]

    def plot_standard(self, sp, norm=True, center=None, center_max=False, **kwargs):
        """
        center can be one of 'Max', 'Left', 'Right', 'Left_fit', 'Right_fit', 'Gauss'
        """

        # Backward compatibility
        if center_max:
            center='Max'

        factor = np.sign(self.charge)
        if norm:
            factor *= self.charge/self.integral

        center_index = None
        if center is None:
            pass
        elif center == 'Max':
            center_index = np.argmax(np.abs(self.current))
        elif center == 'Left':
            center_index = misc.find_rising_flank(self.current)
        elif center == 'Right':
            center_index = len(self.current) - misc.find_rising_flank(self.current[::-1])
        elif center == 'Left_fit':
            dhf = doublehornfit.DoublehornFit(self._xx, self._yy)
            center_index = np.argmin((self._xx - dhf.pos_left)**2)
        elif center == 'Right_fit':
            dhf = doublehornfit.DoublehornFit(self._xx, self._yy)
            center_index = np.argmin((self._xx - dhf.pos_right)**2)
        elif center == 'Gauss':
            center_index = np.argmin((self._xx - self.gaussfit.mean)**2)

        else:
            raise ValueError

        if center_index is None:
            xx = self.time*1e15
        else:
            xx = (self.time - self.time[center_index])*1e15

        if self._yy[0] != 0:
            diff = xx[1] - xx[0]
            x = np.concatenate([[xx[0] - diff], xx])
            y = np.concatenate([[0.], self._yy])
        else:
            x, y = xx, self._yy

        if y[-1] != 0:
            diff = xx[1] - xx[0]
            x = np.concatenate([x, [x[-1] + diff]])
            y = np.concatenate([y, [0.]])

        return sp.plot(x, y*factor/1e3, **kwargs)


def profile_from_blmeas(file_, tt_halfrange, charge, energy_eV, subtract_min=False, zero_crossing=1):
    bl_meas = data_loader.load_blmeas(file_)
    time_meas0 = bl_meas['time_profile%i' % zero_crossing]
    current_meas0 = bl_meas['current%i' % zero_crossing]

    if subtract_min:
        current_meas0 = current_meas0 - current_meas0.min()

    if tt_halfrange is None:
        tt, cc = time_meas0, current_meas0
    else:
        current_meas0 *= charge/current_meas0.sum()
        gf_blmeas = GaussFit(time_meas0, current_meas0)
        time_meas0 -= gf_blmeas.mean

        time_meas1 = np.arange(-tt_halfrange, time_meas0.min(), np.diff(time_meas0).mean())
        time_meas2 = np.arange(time_meas0.max(), tt_halfrange, np.diff(time_meas0).mean())
        time_meas = np.concatenate([time_meas1, time_meas0, time_meas2])
        current_meas = np.concatenate([np.zeros_like(time_meas1), current_meas0, np.zeros_like(time_meas2)])

        tt, cc = time_meas, current_meas
    return BeamProfile(tt, cc, energy_eV, charge)

def dhf_profile(profile):
    dhf = doublehornfit.DoublehornFit(profile.time, profile.current)
    return BeamProfile(dhf.xx, dhf.reconstruction, profile.energy_eV, profile.charge)


#@functools.lru_cache(100)
def get_gaussian_profile(sig_t, tt_halfrange, tt_points, charge, energy_eV, cutoff=1e-3):
    """
    cutoff can be None
    """
    time_arr = np.linspace(-tt_halfrange, tt_halfrange, int(tt_points))
    current_gauss = np.exp(-(time_arr-np.mean(time_arr))**2/(2*sig_t**2))

    if cutoff is not None:
        abs_c = np.abs(current_gauss)
        current_gauss[abs_c<cutoff*abs_c.max()] = 0

    return BeamProfile(time_arr, current_gauss, energy_eV, charge)

def get_average_profile(p_list):
    len_profile = max(len(p) for p in p_list)

    xx_list = [p._xx - p.gaussfit.mean for p in p_list]
    yy_list = [p._yy for p in p_list]

    min_profile = min(x.min() for x in xx_list)
    max_profile = max(x.max() for x in xx_list)

    xx_interp = np.linspace(min_profile, max_profile, len_profile)
    yy_interp_arr = np.zeros([len(p_list), len_profile])

    for n, (xx, yy) in enumerate(zip(xx_list, yy_list)):
        yy_interp_arr[n] = np.interp(xx_interp, xx, yy)

    yy_mean = np.mean(yy_interp_arr, axis=0)
    return xx_interp, yy_mean

class Image:
    def __init__(self, image, x_axis, y_axis, x_unit='m', y_unit='m', subtract_median=False, x_offset=0):

        if x_axis[1] < x_axis[0]:
            x_axis = x_axis[::-1]
            image = image[:,::-1]

        if y_axis[1] < y_axis[0]:
            y_axis = y_axis[::-1]
            image = image[::-1,:]

        if subtract_median:
            image = image - np.median(image)
            np.clip(image, 0, None, out=image)

        self.image = image
        self.x_axis = x_axis - x_offset
        self.y_axis = y_axis
        self.x_unit = x_unit
        self.y_unit = y_unit

    def child(self, new_i, new_x, new_y, x_unit=None, y_unit=None):
        x_unit = self.x_unit if x_unit is None else x_unit
        y_unit = self.y_unit if y_unit is None else y_unit
        return Image(new_i, new_x, new_y, x_unit, y_unit)

    def cut(self, x_min, x_max):
        x_axis = self.x_axis
        x_mask = np.logical_and(x_axis >= x_min, x_axis <= x_max)
        new_image = self.image[:,x_mask]
        new_x_axis = x_axis[x_mask]
        return self.child(new_image, new_x_axis, self.y_axis)

    def reshape_x(self, new_length):
        """
        If new length is larger than current length
        """

        image2 = np.zeros([len(self.y_axis), new_length])
        x_axis2 = np.linspace(self.x_axis.min(), self.x_axis.max(), new_length)
        # Fast interpolation
        delta_x = np.zeros_like(self.image)
        delta_x[:,:-1] = self.image[:,1:] - self.image[:,:-1]
        index_float = (x_axis2 - self.x_axis[0]) / (self.x_axis[1] - self.x_axis[0])
        index = index_float.astype(int)
        index_delta = index_float-index
        np.clip(index, 0, len(self.x_axis)-1, out=index)
        image2 = self.image[:, index] + index_delta * delta_x[:,index]

        image2 = image2 / image2.sum() * self.image.sum()

        return self.child(image2, x_axis2, self.y_axis)

    def slice_x(self, n_slices):
        x_axis, y_axis = self.x_axis, self.y_axis
        max_x_index = len(x_axis) - len(x_axis) % n_slices

        image_extra = np.reshape(self.image[:,:max_x_index], [len(y_axis), n_slices, max_x_index//n_slices])
        new_image = np.mean(image_extra, axis=-1)

        x_axis_reshaped = np.linspace(x_axis[0], x_axis[-1], n_slices)

        output = self.child(new_image, x_axis_reshaped, y_axis)
        return output

    def fit_slice(self, smoothen_first=True, smoothen=100e-6, intensity_cutoff=None, charge=1):
        y_axis = self.y_axis
        n_slices = len(self.x_axis)

        pixelsize = abs(y_axis[1] - y_axis[0])
        smoothen = smoothen/pixelsize

        slice_mean = []
        slice_sigma = []
        slice_gf = []
        for n_slice in range(n_slices):
            intensity = self.image[:,n_slice]
            if smoothen_first:
                yy_conv = gaussian_filter1d(intensity, smoothen)
                gf0 = GaussFit(y_axis, yy_conv, fit_const=False)
                p0 = gf0.popt
            else:
                p0 = None
            gf = GaussFit(y_axis, intensity, fit_const=False, p0=p0)
            slice_mean.append(gf.mean)
            slice_sigma.append(abs(gf.sigma))
            slice_gf.append(gf)

            # Debug bad gaussfits
            #if 101e-15 < self.x_axis[n_slice] < 104e-15:
            #if abs(gf.sigma) < 1e5:
            #    import matplotlib.pyplot as plt
            #    num = plt.gcf().number
            #    plt.figure()
            #    plt.suptitle('Debug 38 fs')
            #    sp = plt.subplot(1,1,1)
            #    gf.plot_data_and_fit(sp)
            #    sp.legend()
            #    plt.figure(num)
            #    plt.show()
            #    import pdb; pdb.set_trace()

        proj = np.sum(self.image, axis=-2)
        proj = proj / np.sum(proj) * charge
        current = proj / (self.x_axis[1] - self.x_axis[0])

        output = {
                'slice_x': self.x_axis,
                'slice_mean': np.array(slice_mean),
                'slice_sigma': np.array(slice_sigma),
                'slice_gf': np.array(slice_gf),
                'slice_intensity': proj,
                'slice_current': current,
                }
        if intensity_cutoff:
            mask = proj > proj.max()*intensity_cutoff
            for key, value in output.items():
                if hasattr(value, 'shape') and value.shape == mask.shape:
                    output[key] = value[mask]
        return output

    def y_to_eV(self, dispersion, energy_eV, ref_y=None):
        if ref_y is None:
            ref_y = GaussFit(self.y_axis, np.sum(self.image, axis=-1)).mean
            #print('y_to_eV', ref_y*1e6, ' [um]')
        E_axis = (self.y_axis-ref_y) * dispersion * energy_eV
        return self.child(self.image, self.x_axis, E_axis, y_unit='eV'), ref_y

    def x_to_t(self, wake_x, wake_time, debug=False, print_=False):
        if wake_time[1] < wake_time[0]:
            wake_x = wake_x[::-1]
            wake_time = wake_time[::-1]

        new_img0 = np.zeros_like(self.image)
        new_t_axis = np.linspace(wake_time.min(), wake_time.max(), self.image.shape[1])
        x_interp = np.interp(new_t_axis, wake_time, wake_x)

        to_print = []
        for t_index, (t, x) in enumerate(zip(new_t_axis, x_interp)):
            x_index = np.argmin((self.x_axis - x)**2)
            new_img0[:,t_index] = self.image[:,x_index]

            if print_:
                to_print.append('%i %i %.1f %.1f' % (t_index, x_index, t*1e15, x*1e6))
        if print_:
            print('\n'.join(to_print))

        diff_x = np.concatenate([np.diff(x_interp), [0]])

        new_img = new_img0 * np.abs(diff_x)
        new_img = new_img / new_img.sum() * self.image.sum()

        output = self.child(new_img, new_t_axis, self.y_axis, x_unit='s')

        if debug:
            ms.figure('Debug x_to_t')
            subplot = ms.subplot_factory(2,3)
            sp_ctr = 1

            sp = subplot(sp_ctr, title='Wake', xlabel='time [fs]', ylabel='Screen x [mm]')
            sp_ctr += 1
            sp.plot(wake_time*1e15, wake_x*1e3)

            sp = subplot(sp_ctr, title='Image projection X', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(self.x_axis*1e3, self.image.sum(axis=-2))

            sp = subplot(sp_ctr, title='Image projection T', xlabel='t [fs]', ylabel='Intensity (arb. units)')
            sp_ctr += 1
            sp.plot(output.x_axis*1e15, output.image.sum(axis=-2))

            sp = subplot(sp_ctr, title='Image old', xlabel='x [mm]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            self.plot_img_and_proj(sp)

            sp = subplot(sp_ctr, title='Image new', xlabel='s [fs]', ylabel='y [mm]', grid=False)
            sp_ctr += 1
            output.plot_img_and_proj(sp)

        return output

    def force_projection(self, proj_x, proj):
        real_proj = np.interp(self.x_axis, proj_x, proj)
        sum_ = self.image.sum(axis=-2)
        sum_[sum_ == 0] = np.inf
        image2 = self.image / sum_ / real_proj.sum() * real_proj
        image2 = image2 / image2.sum() * self.image.sum()

        return self.child(image2, self.x_axis, self.y_axis)

    def plot_img_and_proj(self, sp, x_factor=None, y_factor=None, plot_proj=True, log=True, revert_x=False, plot_gauss=True):

        def unit_to_factor(unit):
            if unit == 'm':
                factor = 1e3
            elif unit == 's':
                factor = 1e15
            elif unit == 'eV':
                factor = 1e-6
            else:
                factor = 1
            return factor

        if x_factor is None:
            x_factor = unit_to_factor(self.x_unit)
        if y_factor is None:
            y_factor = unit_to_factor(self.y_unit)

        x_axis, y_axis, image = self.x_axis, self.y_axis, self.image
        extent = [x_axis[0]*x_factor, x_axis[-1]*x_factor, y_axis[0]*y_factor, y_axis[-1]*y_factor]

        if log:
            image_ = np.clip(image, 1, None)
            log = np.log(image_)
        else:
            log = image

        sp.imshow(log, aspect='auto', extent=extent, origin='lower')
        if plot_proj:
            proj = image.sum(axis=-2)
            proj_plot = (y_axis.min() +(y_axis.max()-y_axis.min()) * proj/proj.max()*0.3)*y_factor
            sp.plot(x_axis*x_factor, proj_plot, color='red')
            gf = GaussFit(x_axis, proj_plot-proj_plot.min(), fit_const=False)
            sp.plot(x_axis*x_factor, gf.reconstruction+proj_plot.min(), color='orange')

            #import matplotlib.pyplot as plt
            #plt.figure()
            #sp = plt.subplot(1,1,1)
            #gf.plot_data_and_fit(sp)
            #plt.show()
            #import pdb; pdb.set_trace()

            proj = image.sum(axis=-1)
            proj_plot = (x_axis.min() +(x_axis.max()-x_axis.min()) * proj/proj.max()*0.3)*x_factor
            sp.plot(proj_plot, y_axis*y_factor, color='red')
            gf = GaussFit(y_axis, proj_plot-proj_plot.min(), fit_const=False)
            sp.plot(gf.reconstruction+proj_plot.min(), y_axis*y_factor, color='orange')

        if revert_x:
            xlim = sp.get_xlim()
            sp.set_xlim(*xlim[::-1])
