import functools
import numpy as np
from scipy.constants import c

import elegant_matrix
import data_loader
from gaussfit import GaussFit
import wf_model

tmp_folder = './'

class Profile:
    normalization = 1

    def compare(self, other):
        xx_min = min(self._xx.min(), other._xx.min())
        xx_max = max(self._xx.max(), other._xx.max())
        xx = np.linspace(xx_min, xx_max, max(len(self._xx), len(other._xx)))
        yy1 = np.interp(xx, self._xx, self._yy, left=0, right=0)
        yy2 = np.interp(xx, other._xx, other._yy, left=0, right=0)
        diff = np.mean((yy1-yy2)**2)
        return diff

    def reshape(self, new_shape):
        _xx = np.linspace(self._xx.min(), self._xx.max(), int(new_shape))
        _yy = np.interp(_xx, self._xx, self._yy)
        self._xx, self._yy = _xx, _yy

    def cutoff(self, cutoff_factor):
        yy = self._yy
        old_sum = np.sum(yy)
        yy[yy<yy.max()*cutoff_factor] = 0
        self._yy = yy / np.sum(yy) * old_sum

    @property
    @functools.lru_cache(1)
    def gaussfit(self):
        return GaussFit(self._xx, self._yy)

class ScreenDistribution(Profile):
    def __init__(self, x, intensity):
        self._xx = x
        self._yy = intensity

    @property
    def x(self):
        return self._xx

    @property
    def intensity(self):
        return self._yy

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

    def wake_effect_on_screen(self, wf_dict, r12):
        wake = wf_dict['dipole']['wake_potential']
        wake_effect = wake/self.energy_eV*r12
        output = {
                't': self.time,
                'x': wake_effect,
                }
        return output


def profile_from_blmeas(file_, tt_halfrange, charge, energy_eV, subtract_min=False):
    bl_meas = data_loader.load_blmeas(file_)
    time_meas0 = bl_meas['time_profile1']
    current_meas0 = bl_meas['current1']
    #energy_eV = bl_meas['energy_eV']

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
    def __init__(self, magnet_file, timestamp, energy_eV='file'):
        self.simulator = elegant_matrix.get_simulator(magnet_file)

        if energy_eV == 'file':
            energy_eV = self.simulator.mag_data.get_prev_datapoint('SARBD01-MBND100:P-SET', timestamp)*1e6
        self.energy_eV = energy_eV
        self.timestamp = timestamp

    def elegant_forward(self, beamProfile, gaps, beam_offsets, struct_lengths, n_bins=200):
        # Generate wakefield

        filenames = []
        for ctr, (gap, beam_offset, struct_length) in enumerate(zip(gaps, beam_offsets, struct_lengths)):
            wf_dict = beamProfile.calc_wake(gap, beam_offset, struct_length)
            filename = tmp_folder+'/streaker%i.sdds' % (ctr+1)
            filenames.append(filename)
            tt = wf_dict['input']['charge_xx']/c
            assert np.all(tt >= 0)

            w_wld = wf_dict['longitudinal_dipole']['single_particle_wake'] # Maybe add quadrupolar wakes later
            w_wxd_deriv = np.zeros_like(tt)
            if beam_offset == 0:
                w_wxd = np.zeros_like(tt)
            else:
                w_wxd = wf_dict['dipole']['single_particle_wake']
            wf_model.write_sdds(filename, tt, w_wld, w_wxd, w_wxd_deriv)

            tt, cc = beamProfile.time, beamProfile.current
            mask = cc != 0
        sim, mat_dict, wf_dicts, disp_dict = self.simulator.simulate_streaker(tt[mask], cc[mask], self.timestamp, 'file', None, self.energy_eV, linearize_twf=False, wf_files=filenames)

        r12_dict = {}
        for n_streaker in (0, 1):
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            r12_dict[n_streaker] = rr[0,1]

        screen_watcher = sim.watch[-1]
        hist, bin_edges0 = np.histogram(screen_watcher['x'], bins=n_bins, density=True)
        screen_xx = (bin_edges0[1:] + bin_edges0[:-1])/2
        screen_hist = hist / hist.max()
        screen = ScreenDistribution(screen_xx, screen_hist)
        #gf = GaussFit(screen_xx, screen_hist)

        forward_dict = {
                'sim': sim,
                'r12_dict': r12_dict,
                'screen_watcher': screen_watcher,
                'screen': screen,
                'beam_profile': beamProfile,
                }

        return forward_dict


    def track_backward(self, forward_dict, forward_dict0, wake_effect):
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        screen_x = forward_dict['screen'].x - forward_dict0['screen'].gaussfit.mean
        screen_intensity = forward_dict['screen'].intensity

        t_interp = np.interp(screen_x, wake_x, wake_time)
        charge_interp = screen_intensity / np.concatenate([np.diff(t_interp), [0]])
        charge_interp[charge_interp == np.inf] = 0
        charge_interp[charge_interp == -np.inf] = 0

        profile0 = forward_dict['beam_profile']

        return BeamProfile(t_interp, charge_interp, profile0.energy_eV, profile0.charge)

    def forward_and_back(self, bp_forward, bp_wake, gaps, beam_offsets, n_streaker, reshape=True):
        track_dict_forward = self.elegant_forward(bp_forward, gaps, beam_offsets, [1, 1])
        track_dict_forward0 = self.elegant_forward(bp_forward, gaps, [0,0], [1, 1])

        wf_dict = bp_wake.calc_wake(gaps[0], beam_offsets[0], 1.)
        wake_effect = bp_wake.wake_effect_on_screen(wf_dict, track_dict_forward0['r12_dict'][n_streaker])
        profile_back = self.track_backward(track_dict_forward, track_dict_forward0, wake_effect)
        if reshape:
            profile_back.reshape(len(bp_forward.time))
        return profile_back


