#import copy
import functools
import numpy as np
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d

import elegant_matrix
import data_loader
from gaussfit import GaussFit
import wf_model

tmp_folder = './'

class Profile:

    def compare(self, other):
        xx_min = min(self._xx.min(), other._xx.min())
        xx_max = max(self._xx.max(), other._xx.max())
        xx = np.linspace(xx_min, xx_max, max(len(self._xx), len(other._xx)))
        yy1 = np.interp(xx, self._xx, self._yy, left=0, right=0)
        yy2 = np.interp(xx, other._xx, other._yy, left=0, right=0)
        diff = (yy1-yy2)**2
        norm = ((yy1+yy2)/2)**2
        return np.sqrt(np.nanmean(diff/norm))
        #return np.nanmean(diff)

    def reshape(self, new_shape):
        if new_shape is None:
            return
        _xx = np.linspace(self._xx.min(), self._xx.max(), int(new_shape))
        _yy = np.interp(_xx, self._xx, self._yy)
        self._xx, self._yy = _xx, _yy

    def cutoff(self, cutoff_factor):
        if cutoff_factor == 0 or cutoff_factor is None:
            return
        yy = self._yy.copy()
        old_sum = np.sum(yy)
        yy[yy<yy.max()*cutoff_factor] = 0
        self._yy = yy / np.sum(yy) * old_sum

    def __len__(self):
        return len(self._xx)

    @property
    @functools.lru_cache(1)
    def gaussfit(self):
        return GaussFit(self._xx, self._yy)

    @property
    def integral(self):
        return np.trapz(self._yy, self._xx)

    def normalize(self, norm=1.):
        self._yy = self._yy / self.integral * norm

    def smoothen(self, gauss_sigma):
        if gauss_sigma is None or gauss_sigma == 0:
            return
        real_sigma = gauss_sigma/np.diff(self._xx).mean()
        new_yy = gaussian_filter1d(self._yy, real_sigma)
        self._yy = new_yy

    def shift(self):
        max_x = self._xx[np.argmax(self._yy)]
        self._xx = self._xx - max_x

    def find_agreement(self, other, step_factor=0.2, max_iter=50):
        if len(self) < len(other):
            self.reshape(len(other))
        elif len(self) > len(other):
            other.reshape(len(self))

        y01 = self._yy
        y02 = other._yy

        x01 = self._xx
        x02 = self._yy

        def min_func(n_shift0):
            n_shift = int(round(n_shift0))

            if n_shift == 0:
                a1, a2 = y01, y02
            elif n_shift > 0:
                a1, a2 = y01[n_shift:], y02[:-n_shift]
            elif n_shift < 0:
                a1, a2 = y01[:n_shift], y02[-n_shift:]

            diff = np.mean((a1-a2)**2) / np.mean(a1+a2)**2

            return diff

        step = 5

        def newton_step(current_shift):
            opt_curr = min_func(current_shift)
            opt_deriv = (min_func(current_shift+step) - opt_curr)/step
            delta = - opt_curr/opt_deriv * step_factor
            new_shift = current_shift + delta
            opt_new = min_func(new_shift)
            return new_shift, opt_curr, opt_new, delta


        shift = 0
        for i in range(max_iter):
            shift, opt_last, opt_new, delta = newton_step(shift)
            if opt_last < opt_new or abs(delta) < 1:
                shift = shift-delta
                break

        n_shift = int(round(shift))

        if n_shift == 0:
            y1, y2 = y01, y02
            x1, x2 = x01, x02
        elif shift > 0:
            y1, y2 = y01[n_shift:], y02[:-n_shift]
            x1, x2 = x01[n_shift:], x02[:-n_shift]
        elif shift < 0:
            y1, y2 = y01[:n_shift], y02[-n_shift:]
            x1, x2 = x01[:n_shift], x02[-n_shift:]

        x1 = x1 - x1.min() + x2.min()
        y2 # for syntax checkers

        self._xx = x1
        self._yy = y1


class ScreenDistribution(Profile):
    def __init__(self, x, intensity, real_x=None):
        self._xx = x
        self._yy = intensity
        self.real_x = real_x

    @property
    def x(self):
        return self._xx

    @property
    def intensity(self):
        return self._yy

def getScreenDistributionFromPoints(x_points, n_bins):
    screen_hist, bin_edges0 = np.histogram(x_points, bins=n_bins, density=True)
    screen_xx = (bin_edges0[1:] + bin_edges0[:-1])/2
    return ScreenDistribution(screen_xx, screen_hist, real_x=x_points)


class BeamProfile(Profile):
    def __init__(self, time, current, energy_eV, charge):
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

    def calc_wake(self, gap, beam_offset, struct_length):
        if (gap, beam_offset, struct_length) in self.wake_dict:
            return self.wake_dict[(gap, beam_offset, struct_length)]

        wf_calc = wf_model.WakeFieldCalculator((self.time - self.time.min())*c, self.current, self.energy_eV, struct_length)
        wf_dict = wf_calc.calc_all(gap/2., R12=0., beam_offset=beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=True)

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
        wake_effect = wake/self.energy_eV*r12
        output = {
                't': self.time,
                'x': wake_effect,
                'charge': self.charge,
                }
        return output


def profile_from_blmeas(file_, tt_halfrange, charge, energy_eV, subtract_min=False):
    bl_meas = data_loader.load_blmeas(file_)
    time_meas0 = bl_meas['time_profile1']
    current_meas0 = bl_meas['current1']

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


def get_gaussian_profile(sig_t, tt_halfrange, tt_points, charge, energy_eV, cutoff=1e-3):
    """
    cutoff can be None
    """
    time_arr = np.linspace(-tt_halfrange, tt_halfrange, int(tt_points))
    current_gauss = np.exp(-(time_arr-np.mean(time_arr))**2/(2*sig_t**2))

    if cutoff is not None:
        current_gauss[current_gauss<cutoff**current_gauss.max()] = 0

    return BeamProfile(time_arr, current_gauss, energy_eV, charge)


class Tracker:
    def __init__(self, magnet_file, timestamp, struct_lengths, energy_eV='file', n_particles=20e3, n_emittances=(300e-9, 300e-9), n_bins=300, screen_cutoff=0.05, smoothen=30e-6, profile_cutoff=0.01, len_screen=1e3):
        self.simulator = elegant_matrix.get_simulator(magnet_file)

        if energy_eV == 'file':
            energy_eV = self.simulator.mag_data.get_prev_datapoint('SARBD01-MBND100:P-SET', timestamp)*1e6
        self.energy_eV = energy_eV
        self.timestamp = timestamp
        self.struct_lengths = struct_lengths
        self.n_particles = n_particles
        self.n_emittances = n_emittances
        self.n_bins = n_bins
        self.screen_cutoff = screen_cutoff
        self.smoothen = smoothen
        self.profile_cutoff = profile_cutoff
        self.len_screen = len_screen

    @functools.lru_cache(1)
    def calcR12(self):
        outp = {}
        for n_streaker in (0, 1):
            mat_dict, _ = self.simulator.get_elegant_matrix(n_streaker, self.timestamp)
            outp[n_streaker] = mat_dict['SARBD02.DSCR050'][0,1]
        return outp

    def get_wake_potential_from_profile(self, time_points, gap, beam_offset, charge):

        wf_hist, bin_edges = np.histogram(time_points, bins=1000)
        wf_hist = wf_hist / np.sum(wf_hist) * charge
        wake_tt = (bin_edges[1:] + bin_edges[:-1])/2
        wake_tt_range = wake_tt.max() - wake_tt.min()
        add_on = np.arange(wake_tt.max(), wake_tt.max() + wake_tt_range*0.1, np.mean(np.diff(wake_tt)))
        wake_tt = np.concatenate([wake_tt, add_on])
        wf_hist = np.concatenate([wf_hist, np.zeros_like(add_on)])
        wake_tt -= wake_tt.min()
        wf_xx = wake_tt*c
        wf_calc = wf_model.WakeFieldCalculator(wf_xx, wf_hist, self.energy_eV, Ls=self.struct_lengths[0])
        wf_dict = wf_calc.calc_all(gap/2., 1., beam_offset, calc_lin_dipole=False, calc_dipole=True, calc_quadrupole=False, calc_long_dipole=False)
        wake = wf_dict['dipole']['wake_potential']

        return wake_tt, wake


    def matrix_forward(self, beamProfile, gaps, beam_offsets):
        mat_dict = self.simulator.get_streaker_matrices(self.timestamp)

        s1 = mat_dict['start_to_s1']

        beta_x = 5.067067
        beta_y = 16.72606
        alpha_x = -0.5774133
        alpha_y = 1.781136

        watch, sim = elegant_matrix.gen_beam(*self.n_emittances, alpha_x, beta_x, alpha_y, beta_y, self.energy_eV, 40e-15, self.n_particles)

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

        for gap, beam_offset in zip(gaps, beam_offsets):
            if beam_offset == 0:
                delta_xp_list.append(0)
            else:
                wake_tt, wake = self.get_wake_potential_from_profile(interp_tt, gap, beam_offset, beamProfile.charge)
                wake_energy = np.interp(beam_before_s1[4,:], wake_tt, wake)
                delta_xp = wake_energy/self.energy_eV
                delta_xp_list.append(delta_xp)


        beam_after_s1 = np.copy(beam_before_s1)
        beam_after_s1[1,:] += delta_xp_list[0]

        beam_before_s2 = np.matmul(mat_dict['s1_to_s2'], beam_after_s1)

        beam_after_s2 = np.copy(beam_before_s2)
        beam_after_s2[1,:] += delta_xp_list[1]

        beam_at_screen = np.matmul(mat_dict['s2_to_screen'], beam_after_s2)
        beam0_at_screen = mat_dict['s2_to_screen'] @ mat_dict['s1_to_s2'] @ beam_before_s1
        beam_at_screen[0] -= beam0_at_screen[0].mean()

        screen = getScreenDistributionFromPoints(beam_at_screen[0,:], n_bins=self.n_bins)

        return {
                'beam_at_screen': beam_at_screen,
                'beam0_at_screen': beam0_at_screen,
                'screen': screen,
                }

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
            sim, mat_dict, wf_dicts, disp_dict = self.simulator.simulate_streaker(tt[mask], cc[mask], self.timestamp, 'file', None, self.energy_eV, linearize_twf=False, wf_files=filenames, n_particles=self.n_particles, n_emittances=self.n_emittances)
        except Exception as e:
            print(e)
            raise
            import pdb; pdb.set_trace()

        r12_dict = {}
        for n_streaker in (0, 1):
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            r12_dict[n_streaker] = rr[0,1]

        screen_watcher = sim.watch[-1]
        screen = getScreenDistributionFromPoints(screen_watcher['x'], self.n_bins)
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

    def track_backward(self, screen, screen0, wake_effect):
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        screen_x = screen.x - screen0.gaussfit.mean
        screen_intensity = screen.intensity

        t_interp = np.interp(screen_x, wake_x, wake_time)
        charge_interp = screen_intensity / np.concatenate([np.diff(t_interp), [0]])
        charge_interp[charge_interp == np.inf] = 0
        charge_interp[charge_interp == -np.inf] = 0
        charge_interp[np.isnan(charge_interp)] = 0

        bp = BeamProfile(t_interp, charge_interp, self.energy_eV, wake_effect['charge'])
        bp.reshape(self.len_screen)
        bp.cutoff(self.profile_cutoff)
        if np.any(np.isnan(charge_interp)):
            raise ValueError('NaNs in beam profile')
        return bp

    def forward_and_back(self, bp_forward, bp_wake, gaps, beam_offsets, n_streaker):
        track_dict_forward = self.elegant_forward(bp_forward, gaps, beam_offsets)
        track_dict_forward0 = self.elegant_forward(bp_forward, gaps, [0,0])

        wf_dict = bp_wake.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], 1.)
        wake_effect = bp_wake.wake_effect_on_screen(wf_dict, track_dict_forward0['r12_dict'][n_streaker])
        bp_back = self.track_backward(track_dict_forward['screen'], track_dict_forward0['screen'], wake_effect)

        return {
                'track_dict_forward': track_dict_forward,
                'track_dict_forward0': track_dict_forward0,
                'wake_effect': wake_effect,
                'bp_back': bp_back,
                }

    def back_and_forward(self, screen, screen0, bp_wake, gaps, beam_offsets, n_streaker, output='Screen'):
        wf_dict = bp_wake.calc_wake(gaps[n_streaker], beam_offsets[n_streaker], 1.)
        r12 = self.calcR12()[n_streaker]
        wake_effect = bp_wake.wake_effect_on_screen(wf_dict, r12)

        bp_back = self.track_backward(screen, screen0, wake_effect)
        bp_back.reshape(len(bp_wake.time))
        track_dict = self.elegant_forward(bp_back, gaps, beam_offsets)

        if output == 'Screen':
            return track_dict['screen']
        elif output == 'Full':
            return track_dict
        else:
            raise ValueError('Invalid value for parameter output:', output)

    def gaussian_baf(self, sig_t, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets, n_streaker, charge):

        bp_gauss = get_gaussian_profile(sig_t, tt_halfrange, self.len_screen, charge, self.energy_eV)
        baf = self.back_and_forward(meas_screen, meas_screen0, bp_gauss, gaps, beam_offsets, n_streaker, output='Full')
        screen = baf['screen']
        #screen.shift()

        #meas_screen_shift = copy.deepcopy(meas_screen)
        #meas_screen_shift.shift()

        diff = screen.compare(meas_screen)

        return {
                'diff': diff,
                'screen': screen,
                'baf': baf,
                'bp_gauss': bp_gauss,
                'bp_reconstructed': baf['beam_profile'],
                }

    def find_best_gauss(self, sig_t_range, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets, n_streaker, charge):
        opt_func_values = []
        opt_func_screens = []
        opt_func_profiles = []

        for sig_t in sig_t_range:
            gbaf = self.gaussian_baf(sig_t, tt_halfrange, meas_screen, meas_screen0, gaps, beam_offsets, n_streaker, charge)
            opt_func_values.append(gbaf['diff'])
            opt_func_screens.append(gbaf['screen'])
            opt_func_profiles.append(gbaf['bp_reconstructed'])

        index_min = np.argmin(opt_func_values)
        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_sig_t = sig_t_range[index_min]

        return {
               'gauss_sigma': best_sig_t,
               'reconstructed_screen': best_screen,
               'reconstructed_profile': best_profile,
               }

