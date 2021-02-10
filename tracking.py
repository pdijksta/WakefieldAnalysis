import copy
import functools
import numpy as np
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d

import elegant_matrix
import data_loader
from gaussfit import GaussFit
import wf_model
import misc
import doublehornfit

tmp_folder = './'

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
        diff = (yy1-yy2)**2
        ## WARNING
        norm = ((yy1+yy2)/2)**1.5
        norm[norm == 0] = np.inf
        return np.sqrt(np.nanmean(diff/norm))

    def reshape(self, new_shape):
        _xx = np.linspace(self._xx.min(), self._xx.max(), int(new_shape))
        old_sum = np.sum(self._yy)

        if new_shape >= len(self._xx):
            _yy = np.interp(_xx, self._xx, self._yy)
        else:
            _yy = np.histogram(self._xx, bins=int(new_shape), weights=self._yy)[0]
            #import pdb; pdb.set_trace()

        _yy *= old_sum / _yy.sum()
        self._xx, self._yy = _xx, _yy



    def cutoff(self, cutoff_factor):
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        yy[yy<yy.max()*cutoff_factor] = 0
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

    def smoothen(self, gauss_sigma):
        if gauss_sigma is None or gauss_sigma == 0:
            return
        real_sigma = gauss_sigma/np.diff(self._xx).mean()
        new_yy = gaussian_filter1d(self._yy, real_sigma)
        self._yy = new_yy

    #def shift(self, where_max=0):
    #    max_x = self._xx[np.argmax(self._yy)]
    #    self._xx = self._xx - max_x + where_max

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
        if self.intensity[0] != 0:
            diff = self.x[1] - self.x[0]
            x = np.concatenate([[self.x[0] - diff], self.x])
            y = np.concatenate([[0.], self.intensity])
        else:
            x, y = self.x, self.intensity

        if x[-1] != 0:
            diff = self.x[1] - self.x[0]
            x = np.concatenate([x, [x[-1] + diff]])
            y = np.concatenate([y, [0.]])

        return sp.plot(x*1e3, y/self.integral, **kwargs)



def getScreenDistributionFromPoints(x_points, screen_bins, smoothen=0):
    if smoothen:
        rand = np.random.randn(len(x_points))
        rand[rand>3] = 3
        rand[rand<-3] = -3
        x_points2 = x_points + rand*smoothen
    else:
        x_points2 = x_points
    screen_hist, bin_edges0 = np.histogram(x_points2, bins=screen_bins, density=True)
    screen_xx = (bin_edges0[1:] + bin_edges0[:-1])/2

    return ScreenDistribution(screen_xx, screen_hist)


class BeamProfile(Profile):
    def __init__(self, time, current, energy_eV, charge):
        super().__init__()

        if np.any(np.isnan(time)):
            raise ValueError('nans in time')
        if np.any(np.isnan(current)):
            raise ValueError('nans in current')

        if np.sum(current) <= 0:
            raise ValueError('Sum of current must be larger than 0')

        self._xx = time
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
        wake_effect = wake/self.energy_eV*r12
        quad_effect = quad/self.energy_eV*r12
        output = {
                't': self.time,
                'x': wake_effect,
                'quad': quad_effect,
                'charge': self.charge,
                }
        return output

    def shift(self, center):
        if center == 'Max':
            center_index = np.argmax(self.current)
        elif center == 'Left':
            center_index = misc.find_rising_flank(self.current)
        elif center == 'Right':
            center_index = len(self.current) - misc.find_rising_flank(self.current[::-1])
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

        if norm:
            factor = self.charge/self.integral
        else:
            factor = 1

        center_index = None
        if center == 'Max':
            center_index = np.argmax(self.current)
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
        else:
            raise ValueError

        if center_index is None:
            xx = self.time*1e15
        else:
            xx = (self.time - self.time[center_index])*1e15

        return sp.plot(xx, self.current*factor, **kwargs)



def profile_from_blmeas(file_, tt_halfrange, charge, energy_eV, subtract_min=False, zero_crossing=1):
    bl_meas = data_loader.load_blmeas(file_)
    time_meas0 = bl_meas['time_profile%i' % zero_crossing]
    current_meas0 = bl_meas['current%i' % zero_crossing]

    if subtract_min:
        current_meas0 -= current_meas0.min()

    if tt_halfrange is None:
        tt, cc = time_meas0, current_meas0
    else:
        current_meas0 *= 200e-12/current_meas0.sum()
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
        current_gauss[current_gauss<cutoff*current_gauss.max()] = 0

    #if np.any(np.isnan(current_gauss)):
    #    import pdb; pdb.set_trace()

    #if np.sum(current_gauss) == 0:
    #    import pdb; pdb.set_trace()


    return BeamProfile(time_arr, current_gauss, energy_eV, charge)


class Tracker:
    def __init__(self, magnet_file, timestamp, struct_lengths, n_particles, n_emittances, screen_bins, screen_cutoff, smoothen, profile_cutoff, len_screen, energy_eV='file', forward_method='matrix', compensate_negative_screen=True, optics0='default', quad_wake=True, bp_smoothen=0):
        self.simulator = elegant_matrix.get_simulator(magnet_file)

        if energy_eV == 'file':
            energy_eV = self.simulator.mag_data.get_prev_datapoint('SARBD01-MBND100:P-SET', timestamp)*1e6
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

        if forward_method == 'matrix':
            self.forward = self.matrix_forward
        elif forward_method == 'elegant':
            self.forward = self.elegant_forward

    @functools.lru_cache(1)
    def calcR12(self):
        outp = {}
        for n_streaker in (0, 1):
            mat_dict, _ = self.simulator.get_elegant_matrix(n_streaker, self.timestamp)
            outp[n_streaker] = mat_dict['SARBD02.DSCR050'][0,1]
        return outp

    def get_wake_potential_from_profile(self, profile, gap, beam_offset):

        mask_curr = profile.current != 0
        wake_tt0 = profile.time[mask_curr]
        wake_tt_range = wake_tt0.max() - wake_tt0.min()
        diff = np.mean(np.diff(wake_tt0))
        try:
            add_on = np.arange(wake_tt0.max(), wake_tt0.max() + wake_tt_range*0.1, diff) + diff
        except:
            raise ValueError
        wake_tt = np.concatenate([wake_tt0, add_on])
        wf_current = np.concatenate([profile.current[mask_curr], np.zeros_like(add_on)])
        wake_tt = wake_tt - wake_tt.min()
        wf_xx = wake_tt*c
        #print(wf_current.sum(), wf_xx.min(), wf_xx.max())
        wf_calc = wf_model.WakeFieldCalculator(wf_xx, wf_current, self.energy_eV, Ls=self.struct_lengths[0])
        wf_dict = wf_calc.calc_all(gap/2., 1., beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=True, calc_long_dipole=False)
        wake = wf_dict['dipole']['wake_potential']
        quad = wf_dict['quadrupole']['wake_potential']

        return wake_tt, wake, quad

    def matrix_forward(self, beamProfile, gaps, beam_offsets, debug=False):
        streaker_matrices = self.simulator.get_streaker_matrices(self.timestamp)
        mat_dict = streaker_matrices['mat_dict']

        s1 = streaker_matrices['start_to_s1']

        if self.optics0 == 'default':
            beta_x = elegant_matrix.simulator.beta_x0
            beta_y = elegant_matrix.simulator.beta_y0
            alpha_x = elegant_matrix.simulator.alpha_x0
            alpha_y = elegant_matrix.simulator.alpha_y0
        else:
            beta_x, beta_y, alpha_x, alpha_y = self.optics0

        watch, sim = elegant_matrix.gen_beam(*self.n_emittances, alpha_x, beta_x, alpha_y, beta_y, self.energy_eV/511e3, 40e-15, self.n_particles)

        curr = beamProfile.current
        tt = beamProfile.time
        integrated_curr = np.cumsum(curr)
        integrated_curr /= integrated_curr.max()

        randoms = np.random.rand(self.n_particles)
        interp_tt = np.interp(randoms, integrated_curr, tt)
        interp_tt -= interp_tt.min()

        p_arr = np.ones_like(watch['x'])*self.energy_eV/511e3
        beam_start = np.array([watch['x'], watch['xp'], watch['y'], watch['yp'], interp_tt, p_arr])
        beam_before_s1 = np.matmul(s1, beam_start)

        delta_xp_list = []
        quad_list = []

        wake_dict = {}
        for n_streaker, (gap, beam_offset) in enumerate(zip(gaps, beam_offsets)):
            if beam_offset == 0:
                delta_xp_list.append(0)
                quad_list.append(0)
            else:
                wake_tt, wake, quad = self.get_wake_potential_from_profile(beamProfile, gap, beam_offset)
                wake_dict[n_streaker] = {'wake_t': wake_tt, 'wake': wake, 'quad': quad}
                wake_energy = np.interp(beam_before_s1[4,:], wake_tt, wake)
                quad_energy = np.interp(beam_before_s1[4,:], wake_tt, quad)
                delta_xp = wake_energy/self.energy_eV
                delta_xp_list.append(delta_xp)
                quad_list.append(quad_energy/self.energy_eV)


        #print('Starting beamsize %.1e' % beam_before_s1[0,:].std())
        beam_after_s1 = np.copy(beam_before_s1)
        beam_after_s1[1,:] += delta_xp_list[0]
        if self.quad_wake:
            beam_after_s1[1,:] += quad_list[0]*(beam_before_s1[0,:]-beam_before_s1[0,:].mean())


        beam_before_s2 = np.matmul(streaker_matrices['s1_to_s2'], beam_after_s1)

        beam_after_s2 = np.copy(beam_before_s2)
        beam_after_s2[1,:] += delta_xp_list[1]
        if self.quad_wake:
            quad_effect = quad_list[1]*(beam_before_s2[0,:]-beam_before_s2[0,:].mean())
            beam_after_s2[1,:] += quad_effect
            #import pdb; pdb.set_trace()

        beam_at_screen = np.matmul(streaker_matrices['s2_to_screen'], beam_after_s2)
        beam0_at_screen = streaker_matrices['s2_to_screen'] @ streaker_matrices['s1_to_s2'] @ beam_before_s1
        beam_at_screen[0] -= beam0_at_screen[0].mean()

        if False and not any(beam_offsets):
            print()
            for label, beam in [
                    ('beam_start', beam_start),
                    ('before_s1', beam_before_s1),
                    ('after_s1', beam_after_s1),
                    ('before_s2', beam_before_s2),
                    ('after_s2', beam_after_s2),
                    ('at_screen', beam_at_screen),
                    ]:

                for indices, dim in [((0,1), 'X'), ((2,3), 'Y')]:
                    pos, angle = indices
                    m11 = np.var(beam[pos,:])
                    m22 = np.var(beam[angle,:])
                    m12 = np.mean((beam[pos,:]-beam[pos,:].mean()) * (beam[angle,:] - beam[angle,:].mean()))
                    emitx = np.sqrt(m11*m22-m12**2)
                    betax = m11/emitx
                    alfax = -m12/emitx
                    beamsize = np.sqrt(emitx*betax)
                    print(label, beamsize, emitx, betax, alfax, dim)
                    if np.isnan(emitx):
                        import pdb; pdb.set_trace()
                print()


        screen = getScreenDistributionFromPoints(beam_at_screen[0,:], self.screen_bins, self.smoothen)
        screen_no_smoothen = getScreenDistributionFromPoints(beam_at_screen[0,:], self.screen_bins, 0)

        #screen.smoothen(self.smoothen)
        screen.cutoff(self.screen_cutoff)
        screen.reshape(self.len_screen)
        screen.normalize()


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

        try:
            sim, mat_dict, wf_dicts, disp_dict = self.simulator.simulate_streaker(tt[mask], cc[mask], self.timestamp, 'file', None, self.energy_eV, linearize_twf=False, wf_files=filenames, n_particles=self.n_particles, n_emittances=self.n_emittances, optics0=self.optics0)
        except Exception as e:
            e
            raise
            #print(e)
            #import pdb; pdb.set_trace()

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

        return forward_dict

    def set_bs_at_streaker(self):
        test_profile = get_gaussian_profile(40e-15, 200e-15, self.len_screen, 200e-12, self.energy_eV)
        forward_dict = self.matrix_forward(test_profile, [10e-3, 10e-3], [0, 0])
        self.bs_at_streaker = forward_dict['bs_at_streaker']


    def track_backward(self, screen, wake_effect, n_streaker, bs_at_streaker=None):
        screen = copy.deepcopy(screen)
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        q_wake_x = wake_effect['quad']
        charge = wake_effect['charge']

        mask_negative = screen.x < 0
        if self.compensate_negative_screen and np.any(mask_negative):
            x_positive = -screen.x[mask_negative][::-1]
            y_positive = screen.intensity[mask_negative][::-1]
            positive_interp = np.interp(screen.x, x_positive, y_positive, left=0, right=0)
            screen_intensity = screen.intensity + positive_interp
            screen_intensity[mask_negative] = 0
            screen._yy = screen_intensity

        if self.quad_wake:
            if bs_at_streaker is None:
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
            bp = BeamProfile(t_interp, charge_interp, self.energy_eV, charge)
        except ValueError as e:
            print(e)
            import matplotlib.pyplot as plt
            import myplotstyle as ms
            ms.figure('')
            sp = plt.subplot(2, 2, 1)
            sp.set_title('Wake')
            sp.plot(wake_time, wake_x, label='Dipole')
            sp.plot(wake_time, q_wake_x*self.bs_at_streaker[n_streaker], label='Quad')
            sp.legend()
            sp = plt.subplot(2, 2, 2)
            sp.set_title('Screen')
            sp.plot(screen.x, screen.intensity)
            plt.show()
            import pdb; pdb.set_trace()
        bp.reshape(self.len_screen)
        bp.cutoff(self.profile_cutoff)
        bp.reshape(self.len_screen)
        if np.any(np.isnan(bp.time)) or np.any(np.isnan(bp.current)):
            raise ValueError('NaNs in beam profile')
        if self.bp_smoothen:
            bp.smoothen(self.bp_smoothen)
        return bp

    def track_backward2(self, screen, profile, gaps, beam_offsets, n_streaker):
        wf_dict = profile.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], self.struct_lengths[n_streaker])
        wake_effect = profile.wake_effect_on_screen(wf_dict, self.calcR12()[n_streaker])
        return self.track_backward(screen, wake_effect, n_streaker)

    def forward_and_back(self, bp_forward, bp_wake, gaps, beam_offsets, n_streaker):

        track_dict_forward = self.forward(bp_forward, gaps, beam_offsets)
        track_dict_forward0 = self.forward(bp_forward, gaps, [0,0])

        screen = track_dict_forward['screen']
        screen0 = track_dict_forward0['screen']
        screen._xx = screen._xx - screen0.gaussfit.mean # Correct for deflection of normal beam

        wf_dict = bp_wake.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], self.struct_lengths[n_streaker])
        wake_effect = bp_wake.wake_effect_on_screen(wf_dict, track_dict_forward0['r12_dict'][n_streaker])
        bp_back = self.track_backward(track_dict_forward['screen'], wake_effect, n_streaker)

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

    def find_best_gauss(self, sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True, details=False):

        opt_func_values = []
        opt_func_screens = []
        opt_func_profiles = []
        gauss_profiles = []
        gauss_wakes = []

        @functools.lru_cache(50)
        def gaussian_baf(sig_t):

            bp_gauss = get_gaussian_profile(sig_t, tt_halfrange, self.len_screen, charge, self.energy_eV)

            if self_consistent:
                bp_back0 = self.track_backward2(meas_screen, bp_gauss, gaps, beam_offsets, n_streaker)
            else:
                bp_back0 = bp_gauss

            baf = self.back_and_forward(meas_screen, bp_back0, gaps, beam_offsets, n_streaker)
            screen = baf['screen']
            diff = screen.compare(meas_screen)
            bp_out = baf['beam_profile']

            opt_func_values.append((float(sig_t), diff))
            opt_func_screens.append(screen)
            opt_func_profiles.append(bp_out)
            gauss_profiles.append(bp_gauss)
            gauss_wakes.append(baf['wake_dict'])

        for sig_t in sig_t_range:
            gaussian_baf(sig_t)

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
            final_screen, final_profile = None, None

        output = {
               'gauss_sigma': best_sig_t,
               'reconstructed_screen': best_screen,
               'reconstructed_profile': best_profile,
               'best_gauss': best_gauss,
               'best_gauss_wake': gauss_wakes[index_min],
               'final_screen': final_screen,
               'final_profile': final_profile,
               }
        if details:
            output.update({
                   'opt_func_values': opt_func_values,
                   'opt_func_screens': opt_func_screens,
                   'opt_func_profiles': opt_func_profiles,
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


