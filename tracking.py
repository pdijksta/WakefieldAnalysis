import matplotlib.pyplot as plt
import copy
import functools
import numpy as np
from scipy.constants import c, m_e, e
from scipy.ndimage import gaussian_filter1d

try:
    from .gaussfit import GaussFit
    from . import elegant_matrix
    from . import data_loader
    from . import wf_model
    from . import misc
    from . import doublehornfit
except ImportError:
    from gaussfit import GaussFit
    import elegant_matrix
    import data_loader
    import wf_model
    import misc
    import doublehornfit


import myplotstyle as ms

tmp_folder = './'

e0_eV = m_e*c**2/e

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
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        abs_yy = np.abs(yy)
        yy[abs_yy<abs_yy.max()*cutoff_factor] = 0
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
    def __init__(self, x, intensity, real_x=None):
        super().__init__()
        self._xx = x
        assert np.all(np.diff(self._xx)>=0)
        self._yy = intensity
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
        center can be one of 'Max', 'Left', 'Right'
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


@functools.lru_cache(100)
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


class Tracker:
    def __init__(self, magnet_file='', timestamp=0, struct_lengths=(1, 1), n_particles=1, n_emittances=(1, 1), screen_bins=0, screen_cutoff=0, smoothen=0, profile_cutoff=0, len_screen=0, energy_eV='file', forward_method='matrix', compensate_negative_screen=True, optics0='default', quad_wake=True, bp_smoothen=0, override_quad_beamsize=False, quad_x_beamsize=(0., 0.)):
        self.simulator = elegant_matrix.get_simulator(magnet_file)

        if energy_eV == 'file':
            energy_eV = self.simulator.get_data('SARBD01-MBND100:P-SET', timestamp)*1e6
        self.energy_eV = energy_eV
        self.timestamp = timestamp
        self.struct_lengths = struct_lengths
        self.n_particles = n_particles
        self.n_emittances = n_emittances
        self.screen_bins = screen_bins
        self.screen_cutoff = screen_cutoff
        self.smoothen = smoothen
        self.profile_cutoff = profile_cutoff
        self.len_screen = len_screen
        self.compensate_negative_screen = compensate_negative_screen
        self.optics0 = optics0
        self.quad_wake = quad_wake
        self.bs_at_streaker = None
        self.bp_smoothen = bp_smoothen
        self.override_quad_beamsize = override_quad_beamsize
        self.quad_x_beamsize = quad_x_beamsize

        self.wake2d = False
        self.hist_bins_2d = (500, 500)
        self.split_streaker = 0

        if forward_method == 'matrix':
            self.forward = self.matrix_forward
        elif forward_method == 'elegant':
            self.forward = self.elegant_forward

    @functools.lru_cache(1)
    def calcR12(self):
        outp = {}
        for n_streaker in (0, 1):
            mat_dict, _ = self.simulator.get_elegant_matrix(int(n_streaker), int(self.timestamp))
            outp[n_streaker] = mat_dict['SARBD02.DSCR050'][0,1]
        return outp

    def get_wake_potential_from_profile(self, profile, gap, beam_offset, n_streaker):

        mask_curr = profile.current != 0
        where = np.argwhere(mask_curr)
        wake_tt0 = profile.time[where[0,0]:where[-1,0]]
        wake_tt_range = wake_tt0.max() - wake_tt0.min()
        diff = np.mean(np.diff(wake_tt0))
        try:
            add_on = np.arange(wake_tt0.max(), wake_tt0.max() + wake_tt_range*0.1, diff) + diff
        except:
            raise ValueError
        wake_tt = np.concatenate([wake_tt0, add_on])
        wf_current = np.concatenate([profile.current[where[0,0]:where[-1,0]], np.zeros_like(add_on)])
        wake_tt = wake_tt - wake_tt.min()
        wf_xx = wake_tt*c
        #print(wf_current.sum(), wf_xx.min(), wf_xx.max())
        wf_calc = wf_model.WakeFieldCalculator(wf_xx, wf_current, self.energy_eV, Ls=self.struct_lengths[n_streaker])
        wf_dict = wf_calc.calc_all(gap/2., 1., beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=self.quad_wake, calc_long_dipole=False)
        wake = wf_dict['dipole']['wake_potential']
        if self.quad_wake:
            quad = wf_dict['quadrupole']['wake_potential']
        else:
            quad = 0

        return wake_tt, wake, quad

    def matrix_forward(self, beamProfile, gaps, beam_offsets):

        # wake_dict is needed in output for diagnostics
        wake_dict = {}

        def calc_wake(n_streaker, beam_before):
            """
            Calculate changes in xp for every particle.
            Optionally calculate quadrupole wake.
            """
            beam_offset = beam_offsets[n_streaker]
            gap = gaps[n_streaker]

            if beam_offset == 0:
                dipole = 0
                quad = 0
                wake_tt = 0
                dipole_wake = 0
                quad_wake = 0
            else:
                wake_tt, dipole_wake, quad_wake = self.get_wake_potential_from_profile(beamProfile, gap, beam_offset, n_streaker)
                wake_energy = np.interp(beam_before[4,:], wake_tt, dipole_wake)
                dipole = wake_energy/self.energy_eV*np.sign(beamProfile.charge)
                if self.quad_wake:
                    quad_energy = np.interp(beam_before[4,:], wake_tt, quad_wake)
                    quad0 = quad_energy/self.energy_eV*np.sign(beamProfile.charge)

                    if self.override_quad_beamsize:
                        quad_x_factor = self.quad_x_beamsize[n_streaker] / beam_before[0,:].std()
                    else:
                        quad_x_factor = 1.
                    quad_x = beam_before[0,:]-beam_before[0,:].mean()
                    quad = quad0 * quad_x * quad_x_factor
                else:
                    quad = 0

            wake_dict[n_streaker] = {
                'wake_t': wake_tt,
                'wake': dipole_wake,
                'quad': quad_wake,
                'delta_xp_dipole': dipole,
                'delta_xp_quad': quad,
            }

            #print('1d Dipole quad mean', np.abs(dipole).mean(), np.abs(quad).mean())
            return dipole, quad

        def calc_wake2d(n_streaker, beam_before):
            """
            Calculate changes in xp for every particle using a 2d histogram of particles.
            Equations are evaluated based on the x of every histogram point, instead of assuming the same x for every particle.
            Optionally calculate quadrupole wake.
            """
            beam_offset = beam_offsets[n_streaker]
            gap = gaps[n_streaker]

            # Careful! Effects from beamsize are not considered when beam_offset is 0
            if beam_offset == 0:
                return 0, 0

            if self.override_quad_beamsize:
                raise ValueError('Override quad beamsize not tested for 2d wake')
                x_coords0 = beam_before[0,:]
                x_coords0 = x_coords0.mean() + (x_coords0 - x_coords0.mean())*self.quad_x_beamsize[n_streaker] / x_coords0.std()
                x_coords = x_coords0 + beam_offset
            else:
                x_coords = beam_before[0,:]+beam_offset

            dict_dipole_2d = wf_model.wf2d(beam_before[4,:]*c, x_coords, gap/2., beamProfile.charge, wf_model.wxd, self.hist_bins_2d)
            dipole = dict_dipole_2d['wake_on_particles']/self.energy_eV

            if self.quad_wake:
                dict_quad_2d = wf_model.wf2d_quad(beam_before[4,:]*c, beam_before[0,:]+beam_offset, gap/2., beamProfile.charge, wf_model.wxq, self.hist_bins_2d)

                quad = dict_quad_2d['wake_on_particles']/self.energy_eV
                quad_wake = dict_quad_2d['wake']
            else:
                quad = 0
                quad_wake = 0

            wake_dict[n_streaker] = {
                'wake_t': dict_dipole_2d['s_bins']/c,
                'wake': dict_dipole_2d['wake'],
                'quad': quad_wake,
                'delta_xp_dipole': dipole,
                'delta_xp_quad': quad,
            }

            #print('2d Dipole quad mean', np.abs(dipole).mean(), np.abs(quad).mean())
            return dipole, quad

        def split_streaker(n_streaker, beam_before):
            if self.split_streaker in (0, 1):
                beam_after = np.copy(beam_before)
                if self.wake2d:
                    dipole, quad = calc_wake2d(n_streaker, beam_before)
                else:
                    dipole, quad = calc_wake(n_streaker, beam_before)
                beam_after[1,:] += dipole + quad
                #print('Comb Dipole quad mean', np.abs(dipole+quad).mean())
            else:
                drift_minus = misc.drift(-self.struct_lengths[n_streaker]/2)
                beam_after = beam_after = np.matmul(drift_minus, beam_before)
                delta_l = self.struct_lengths[n_streaker]/self.split_streaker/2
                drift = misc.drift(delta_l)
                comb_effect = 0

                for n_split in range(self.split_streaker):
                    np.matmul(drift, beam_after, out=beam_after)

                    if self.wake2d:
                        dipole, quad = calc_wake2d(n_streaker, beam_after)
                    else:
                        dipole, quad = calc_wake(n_streaker, beam_after)
                    beam_after[1,:] += (dipole + quad)/self.split_streaker
                    comb_effect += (dipole + quad)/self.split_streaker

                    #if beam_offsets[n_streaker] == 0.004692:
                    #    import pickle
                    #    with open('./beam_after.pkl', 'wb') as f:
                    #        pickle.dump(beam_after, f)

                    np.matmul(drift, beam_after, out=beam_after)
                np.matmul(drift_minus, beam_after, out=beam_after)
                #print('Comb Dipole quad mean', np.abs(comb_effect).mean())
            return beam_after


        ## Obtain first order matrices from elegant

        streaker_matrices = self.simulator.get_streaker_matrices(self.timestamp)
        mat_dict = streaker_matrices['mat_dict']

        ## Generate beam

        # Optics
        if self.optics0 == 'default':
            beta_x = elegant_matrix.simulator.beta_x0
            beta_y = elegant_matrix.simulator.beta_y0
            alpha_x = elegant_matrix.simulator.alpha_x0
            alpha_y = elegant_matrix.simulator.alpha_y0
        else:
            beta_x, beta_y, alpha_x, alpha_y = self.optics0

        watch, sim = elegant_matrix.gen_beam(*self.n_emittances, alpha_x, beta_x, alpha_y, beta_y, self.energy_eV/e0_eV, 40e-15, self.n_particles)

        # Longitudinal positions according to input current profile
        curr = beamProfile.current
        tt = beamProfile.time
        integrated_curr = np.cumsum(curr)
        integrated_curr /= integrated_curr[-1]

        randoms = np.random.rand(self.n_particles)
        interp_tt = np.interp(randoms, integrated_curr, tt)
        interp_tt -= interp_tt.min()

        p_arr = np.ones_like(watch['x'])*self.energy_eV/e0_eV
        beam_start = np.array([watch['x'], watch['xp'], watch['y'], watch['yp'], interp_tt, p_arr])

        ## Propagation of initial beam

        s1 = streaker_matrices['start_to_s1']
        beam_before_s1 = np.matmul(s1, beam_start)
        beam_after_s1 = split_streaker(0, beam_before_s1)

        beam_before_s2 = np.matmul(streaker_matrices['s1_to_s2'], beam_after_s1)
        beam_after_s2 = split_streaker(1, beam_before_s2)

        #if beam_offsets[1] != 0:
        #    import pickle
        #    filename = './investigate_streaking.pkl'
        #    with open(filename, 'wb') as f:
        #        pickle.dump({
        #            'beam_before_s2': beam_before_s2,
        #            'delta_xp': delta_xp_list[1],
        #            'beam_offset': beam_offsets[1],
        #            'gap': gaps[1],
        #            'profile': beamProfile,
        #            }, f)
        #    print('Saved %s' % filename)
        #    import sys; sys.exit()


        beam_at_screen = np.matmul(streaker_matrices['s2_to_screen'], beam_after_s2)
        beam0_at_screen = streaker_matrices['s2_to_screen'] @ streaker_matrices['s1_to_s2'] @ beam_before_s1
        beam_at_screen[0] -= beam0_at_screen[0].mean()

        #if not any(beam_offsets):
        #    print()
        #    for label, beam in [
        #            ('beam_start', beam_start),
        #            ('before_s1', beam_before_s1),
        #            ('after_s1', beam_after_s1),
        #            ('before_s2', beam_before_s2),
        #            ('after_s2', beam_after_s2),
        #            ('at_screen', beam_at_screen),
        #            ]:

        #        for indices, dim in [((0,1), 'X'), ((2,3), 'Y')]:
        #            pos, angle = indices
        #            m11 = np.var(beam[pos,:])
        #            m22 = np.var(beam[angle,:])
        #            m12 = np.mean((beam[pos,:]-beam[pos,:].mean()) * (beam[angle,:] - beam[angle,:].mean()))
        #            emitx = np.sqrt(m11*m22-m12**2)
        #            betax = m11/emitx
        #            alfax = -m12/emitx
        #            beamsize = np.sqrt(emitx*betax)
        #            print(label, beamsize, emitx, betax, alfax, dim)
        #            if np.isnan(emitx):
        #                raise ValueError
        #                #import pdb; pdb.set_trace()
        #        print()


        screen = getScreenDistributionFromPoints(beam_at_screen[0,:], self.screen_bins, 0)
        screen_no_smoothen = copy.deepcopy(screen)
        screen.smoothen(self.smoothen)

        for s_ in (screen_no_smoothen, screen):
            s_.reshape(self.len_screen)
            s_.cutoff(self.screen_cutoff)
            s_.normalize()

        r12_dict = {}
        for n_streaker in (0, 1):
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            r12_dict[n_streaker] = rr[0,1]

        output = {
                'beam_at_screen': beam_at_screen,
                'beam0_at_screen': beam0_at_screen,
                'screen': screen,
                'screen_no_smoothen': screen_no_smoothen,
                'beam_profile': beamProfile,
                'r12_dict': r12_dict,
                'initial_beam': watch,
                'wake_dict': wake_dict,
                'bs_at_streaker': [beam_before_s1[0,:].std(), beam_before_s2[0,:].std()],
                }

        return output

    def elegant_forward(self, beamProfile, gaps, beam_offsets):
        # Generate wakefield

        filenames = []
        sdds_wakes = []
        for ctr, (gap, beam_offset, struct_length) in enumerate(zip(gaps, beam_offsets, self.struct_lengths)):
            filename = tmp_folder+'/streaker%i.sdds' % (ctr+1)
            filenames.append(filename)
            sdds_wake = beamProfile.write_sdds(filename, gap, beam_offset, struct_length)
            sdds_wakes.append(sdds_wake)

        tt, cc = beamProfile.time, beamProfile.current
        mask = cc != 0

        sim, mat_dict, wf_dicts, disp_dict = self.simulator.simulate_streaker(tt[mask], cc[mask], self.timestamp, 'file', None, self.energy_eV, wf_files=filenames, n_particles=self.n_particles, n_emittances=self.n_emittances, optics0=self.optics0)

        r12_dict = {}
        for n_streaker in (0, 1):
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            r12_dict[n_streaker] = rr[0,1]

        screen_watcher = sim.watch[-1]
        screen = getScreenDistributionFromPoints(screen_watcher['x'], self.screen_bins)
        screen.smoothen(self.smoothen)
        screen.cutoff(self.screen_cutoff)
        screen.reshape(self.len_screen)
        screen.normalize()

        forward_dict = {
                'sim': sim,
                'r12_dict': r12_dict,
                'screen_watcher': screen_watcher,
                'screen': screen,
                'beam_profile': beamProfile,
                'sdds_wakes': sdds_wakes,
                }

        #output = {
        #        'beam_at_screen': beam_at_screen,
        #        'beam0_at_screen': beam0_at_screen,
        #        'screen': screen,
        #        'screen_no_smoothen': screen_no_smoothen,
        #        'beam_profile': beamProfile,
        #        'r12_dict': r12_dict,
        #        'initial_beam': watch,
        #        'wake_dict': wake_dict,
        #        'bs_at_streaker': [beam_before_s1[0,:].std(), beam_before_s2[0,:].std()],
        #        }


        return forward_dict

    def set_bs_at_streaker(self):
        test_profile = get_gaussian_profile(40e-15, 200e-15, self.len_screen, 200e-12, self.energy_eV)
        forward_dict = self.matrix_forward(test_profile, [10e-3, 10e-3], [0, 0])
        self.bs_at_streaker = forward_dict['bs_at_streaker']


    def track_backward(self, screen, wake_effect, n_streaker, plot_details=False):
        screen = copy.deepcopy(screen)
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        q_wake_x = wake_effect['quad']
        charge = wake_effect['charge']

        if np.any(np.diff(wake_x) < 0):
            assert np.all(np.diff(wake_x) <= 0)
            wake_x = wake_x[::-1]
            q_wake_x = q_wake_x[::-1]
            wake_time = wake_time[::-1]
        elif np.all(np.diff(wake_x) >= 0):
            pass
        else:
            raise ValueError

        if abs(wake_x.max()) > abs(wake_x.min()):
            mask_negative = screen.x < 0
        else:
            mask_negative = screen.x > 0

        if self.compensate_negative_screen and np.any(mask_negative):
            x_positive = -screen.x[mask_negative][::-1]
            y_positive = screen.intensity[mask_negative][::-1]
            if np.all(np.diff(x_positive) < 0):
                x_positive = x_positive[::-1]
                y_positive = y_positive[::-1]
            positive_interp = np.interp(screen.x, x_positive, y_positive, left=0, right=0)
            screen_intensity = screen.intensity + positive_interp
            screen_intensity[mask_negative] = 0
            screen._yy = screen_intensity

        if self.quad_wake:
            if self.override_quad_beamsize:
                bs_at_streaker = self.quad_x_beamsize[n_streaker]
            else:
                if self.bs_at_streaker is None:
                    self.set_bs_at_streaker()
                bs_at_streaker = self.bs_at_streaker[n_streaker]

            screen.reshape(self.n_particles)
            rand0 = np.random.randn(len(screen.x))
            rand0[rand0>3] = 3
            rand0[rand0<-3] = -3
            randx = rand0*bs_at_streaker
            t_interp0 = np.zeros_like(screen.x)

            for n_x, (x, rx) in enumerate(zip(screen.x, randx)):
                t_interp0[n_x] = np.interp(x, wake_x+rx*q_wake_x, wake_time)

            charge_interp, hist_edges2 = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
            charge_interp[0] = 0
            charge_interp[-1] = 0
            t_interp = hist_edges2[1:]

        else:
            screen.reshape(self.n_particles)
            t_interp0 = np.interp(screen.x, wake_x, wake_time)
            charge_interp, hist_edges = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
            charge_interp[0] = 0
            charge_interp[-1] = 0
            t_interp = np.linspace(t_interp0[0], t_interp0[-1], len(charge_interp))

        try:
            assert np.all(np.diff(t_interp) >= 0)
            bp = BeamProfile(t_interp, charge_interp, self.energy_eV, charge)
        except ValueError as e:
            print(e)
            ms.figure('')
            self.set_bs_at_streaker()
            subplot = ms.subplot_factory(2,2)
            sp = subplot(1, title='Wake', xlabel='t', ylabel='$\Delta$ x')
            sp.plot(wake_time, wake_x, label='Dipole')
            sp.plot(wake_time, q_wake_x*self.bs_at_streaker[n_streaker], label='Quad')
            sp.legend()
            sp = subplot(2, title='Screen', xlabel='x')
            sp.plot(screen.x, screen.intensity)

            sp = subplot(3, title='Current profile', xlabel='time', ylabel='Current')
            sp.plot(t_interp, charge_interp)
            plt.show()
            #import pdb; pdb.set_trace()
        bp.reshape(self.len_screen)
        bp.cutoff(self.profile_cutoff)
        bp.crop()
        if np.any(np.isnan(bp.time)) or np.any(np.isnan(bp.current)):
            #import pdb; pdb.set_trace()
            raise ValueError('NaNs in beam profile')
        if self.bp_smoothen:
            #bp0 = copy.deepcopy(bp)
            bp.smoothen(self.bp_smoothen)
            #bp1 = copy.deepcopy(bp)
            bp.reshape(self.len_screen)
            #import pdb; pdb.set_trace()


        if plot_details:
            ms.figure('track_backward')
            subplot = ms.subplot_factory(2,2)
            sp_wake = subplot(1, title='Wake effect', xlabel='t [fs]', ylabel='$\Delta$ x [mm]')
            sp_wake.plot(wake_time*1e15, wake_x*1e3)

            sp_screen = subplot(2, title='Screen dist', xlabel='x [mm]', ylabel='Intensity (arb. units)')
            screen.plot_standard(sp_screen)

            sp_profile = subplot(3, title='Interpolated profile', xlabel='t [fs]', ylabel='Current [kA]')
            bp.plot_standard(sp_profile)

        return bp

    def track_backward2(self, screen, profile, gaps, beam_offsets, n_streaker, **kwargs):
        wf_dict = profile.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], self.struct_lengths[n_streaker])
        wake_effect = profile.wake_effect_on_screen(wf_dict, self.calcR12()[n_streaker])
        return self.track_backward(screen, wake_effect, n_streaker, **kwargs)

    def forward_and_back(self, bp_forward, bp_wake, gaps, beam_offsets, n_streaker):

        track_dict_forward = self.forward(bp_forward, gaps, beam_offsets)
        track_dict_forward0 = self.forward(bp_forward, gaps, [0,0])

        #screen_f0 = copy.deepcopy(track_dict_forward['screen'])
        screen = track_dict_forward['screen']
        screen0 = track_dict_forward0['screen']
        screen._xx = screen._xx - screen0.gaussfit.mean # Correct for deflection of normal beam

        wf_dict = bp_wake.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], self.struct_lengths[n_streaker])
        wake_effect = bp_wake.wake_effect_on_screen(wf_dict, track_dict_forward0['r12_dict'][n_streaker])
        bp_back = self.track_backward(screen, wake_effect, n_streaker)

        #ms.figure('forward_and_back')
        #subplot = ms.subplot_factory(2,2)
        #sp_ctr = 1
        #sp = subplot(sp_ctr, title='Screens')
        #sp_ctr += 1
        #sp.plot(screen._xx, screen._yy, label='Input')
        #screen = screen_f0
        #sp.plot(screen._xx, screen._yy, label='Forward')
        #screen = screen0
        #sp.plot(screen._xx, screen._yy, label='Forward 0')
        #sp.legend()

        #sp = subplot(sp_ctr, title='Profiles')
        #p = bp_forward
        #sp.plot(p._xx, p._yy, label='Input')
        #p = bp_back
        #sp.plot(p._xx, p._yy, label='Back')

        #sp.legend()

        #plt.show()

        return {
                'track_dict_forward': track_dict_forward,
                'track_dict_forward0': track_dict_forward0,
                'wake_effect': wake_effect,
                'bp_back': bp_back,
                }

    def back_and_forward(self, screen, bp_wake, gaps, beam_offsets, n_streaker):

        bp_back = self.track_backward2(screen, bp_wake, gaps, beam_offsets, n_streaker)
        track_dict = self.forward(bp_back, gaps, beam_offsets)

        return track_dict

    ## Below are optimization methods to find a good beam profile

    def iterate(self, meas_screen, profile, gaps, beam_offsets, n_streaker, n_iter):
        """
        Does  not really work. The reconstructed profile drifts away.
        """

        profiles, screens, diffs = [], [], []

        for n in range(n_iter):
            baf = self.back_and_forward(meas_screen, profile, gaps, beam_offsets, n_streaker)
            profile = baf['beam_profile']
            recon_screen = baf['screen']
            diff = recon_screen.compare(meas_screen)

            profiles.append(profile)
            screens.append(recon_screen)
            diffs.append(diff)

        diffs = np.array(diffs)
        best_index = np.argmin(diffs)

        return {
                'profiles': profiles,
                'screens': screens,
                'diffs': diffs,
                'best_screen': screens[best_index],
                'best_profile': profiles[best_index],
                }

    def find_best_gauss(self, sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True, details=True):

        opt_func_values = []
        opt_func_screens = []
        opt_func_screens_no_smoothen = []
        opt_func_profiles = []
        opt_func_sigmas = []
        gauss_profiles = []
        gauss_wakes = []

        meas_screen.reshape(self.len_screen)
        meas_screen.cutoff(self.screen_cutoff)
        meas_screen.crop()
        meas_screen.reshape(self.len_screen)


        @functools.lru_cache(50)
        def gaussian_baf(sig_t):

            assert 1e-15 < sig_t < 1e-12

            bp_gauss = get_gaussian_profile(sig_t, float(tt_halfrange), int(self.len_screen), float(charge), float(self.energy_eV))

            if self_consistent:
                bp_back0 = self.track_backward2(meas_screen, bp_gauss, gaps, beam_offsets, n_streaker)
            else:
                bp_back0 = bp_gauss

            baf = self.back_and_forward(meas_screen, bp_back0, gaps, beam_offsets, n_streaker)
            screen = copy.deepcopy(baf['screen'])
            diff = screen.compare(meas_screen)
            bp_out = baf['beam_profile']

            opt_func_values.append((float(sig_t), diff))
            opt_func_screens.append(screen)
            opt_func_profiles.append(bp_out)
            opt_func_sigmas.append(sig_t)
            opt_func_screens_no_smoothen.append(baf['screen_no_smoothen'])
            gauss_profiles.append(bp_gauss)
            gauss_wakes.append(baf['wake_dict'])

        for sig_t in sig_t_range:
            gaussian_baf(float(sig_t))

        opt_func_values = np.array(opt_func_values)

        index_min = np.argmin(opt_func_values[:, 1])
        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_sig_t = sig_t_range[index_min]
        best_gauss = gauss_profiles[index_min]

        # Final step
        if not self_consistent:
            baf = self.back_and_forward(meas_screen, best_profile, gaps, beam_offsets, n_streaker)
            final_screen = baf['screen']
            final_profile = baf['beam_profile']
        else:
            final_screen, final_profile = best_screen, best_profile

        output = {
               'gauss_sigma': best_sig_t,
               'reconstructed_screen': best_screen,
               'reconstructed_screen_no_smoothen': opt_func_screens_no_smoothen[index_min],
               'reconstructed_profile': best_profile,
               'best_gauss': best_gauss,
               'best_gauss_wake': gauss_wakes[index_min],
               'final_screen': final_screen,
               'final_profile': final_profile,
               'meas_screen': meas_screen,
               }
        if details:
            output.update({
                   'opt_func_values': opt_func_values,
                   'opt_func_screens': opt_func_screens,
                   'opt_func_profiles': opt_func_profiles,
                   'opt_func_sigmas': np.array(opt_func_sigmas),
                   'opt_func_wakes': gauss_wakes,
                   })

        return output

    def scale_existing_profile(self, scale_range, profile, meas_screen, gaps, beam_offsets, n_streaker):

        opt_func_values = []
        opt_func_screens = []
        opt_func_profiles = []
        scaled_profiles = []

        @functools.lru_cache(50)
        def gaussian_baf(scale_factor):

            bp_scaled = copy.deepcopy(profile)
            bp_scaled.scale_xx(scale_factor)
            scaled_profiles.append(bp_scaled)

            bp_back0 = self.track_backward2(meas_screen, bp_scaled, gaps, beam_offsets, n_streaker)

            baf = self.back_and_forward(meas_screen, bp_back0, gaps, beam_offsets, n_streaker)
            screen = baf['screen']
            diff = screen.compare(meas_screen)
            bp_out = baf['beam_profile']

            opt_func_values.append(diff)
            opt_func_screens.append(screen)
            opt_func_profiles.append(bp_out)

        for sig_t in scale_range:
            gaussian_baf(sig_t)

        index_min = np.argmin(opt_func_values)
        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_scale = scale_range[index_min]

        return {
               'scale': best_scale,
               'reconstructed_screen': best_screen,
               'reconstructed_profile': best_profile,
               'scaled_profiles': scaled_profiles,
               'all_profiles': opt_func_profiles,
               'all_screens': opt_func_screens,
               }

