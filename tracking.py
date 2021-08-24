import matplotlib.pyplot as plt
import bisect
import copy
import functools
import numpy as np
from scipy.constants import c, m_e, e

try:
    from . import elegant_matrix
    from . import wf_model
    from . import misc2 as misc
    from . import image_and_profile as iap
    from . import myplotstyle as ms
    from . import config
except ImportError:
    import elegant_matrix
    import wf_model
    import misc2 as misc
    import image_and_profile as iap
    import myplotstyle as ms
    import config

tmp_folder = './'
e0_eV = m_e*c**2/e

# Backward compatibility
BeamProfile = iap.BeamProfile
ScreenDistribution = iap.ScreenDistribution

class Tracker:
    def __init__(self, magnet_file='', timestamp=0, struct_lengths=(1, 1), n_particles=1, n_emittances=(1, 1), screen_bins=0, screen_cutoff=0, smoothen=0, profile_cutoff=0, len_screen=0, energy_eV='file', forward_method='matrix', compensate_negative_screen=True, optics0='default', quad_wake=True, bp_smoothen=0, override_quad_beamsize=False, quad_x_beamsize=(0., 0.), quad_wake_back=False, beamline='Aramis'):

        self._r12 = None
        self._disp = None
        self.beamline = beamline
        if magnet_file:
            self.set_simulator(magnet_file, energy_eV, timestamp)

        self._timestamp = timestamp
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
        self.quad_wake_back = quad_wake_back
        self.bs_at_streaker = None
        self.bp_smoothen = bp_smoothen
        self.override_quad_beamsize = override_quad_beamsize
        self.quad_x_beamsize = quad_x_beamsize

        self.wake2d = False
        self.hist_bins_2d = (500, 500)
        self.split_streaker = 0

        self.gauss_prec = 0.1e-15

        if forward_method == 'matrix':
            self.forward = self.matrix_forward
        elif forward_method == 'elegant':
            self.forward = self.elegant_forward

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, t):
        self._timestamp = t
        self._r12 = None
        self._disp = None

    def set_simulator(self, magnet_file, energy_eV='file', timestamp=None):
        self.simulator = elegant_matrix.get_simulator(magnet_file, self.beamline)
        if energy_eV == 'file':
            energy_pv = config.beamline_energypv[self.beamline]
            self.energy_eV = self.simulator.get_data(energy_pv, timestamp)*1e6
            #try:
            #    self.energy_eV = self.simulator.get_data('SARBD01-MBND100:ENERGY-OP', timestamp)*1e6
            #except KeyError:
            #    self.energy_eV = self.simulator.get_data('SARBD01-MBND100:P-SET', timestamp)*1e6
        else:
            self.energy_eV = energy_eV
        self._r12 is None
        self._disp is None

    def calcR12(self):
        self.calcDisp()
        return self._r12

    @functools.lru_cache(1)
    def calcDisp(self):
        if self._disp is None or self._r12 is None:
            outp_disp = {}
            outp_r12 = {}
            screen = config.beamline_screens[self.beamline].replace('-', '.')
            for n_streaker in (0, 1):
                mat_dict, disp_dict = self.simulator.get_elegant_matrix(int(n_streaker), self.timestamp, branch=self.beamline)
                outp_disp[n_streaker] = disp_dict[screen]
                outp_r12[n_streaker] = mat_dict[screen][0,1]
            self._disp = outp_disp
            self._r12 = outp_r12
        return self._disp

    def fit_emittance(self, target_beamsize, assumed_screen_res, tt_halfrange):
        if target_beamsize <= assumed_screen_res:
            raise ValueError('Target beamsize must be larger assumed screen resolution')

        bp_test = iap.get_gaussian_profile(40e-15, tt_halfrange, self.len_screen, 200e-12, self.energy_eV)
        screen_sim = self.matrix_forward(bp_test, [10e-3, 10e-3], [0, 0])['screen_no_smoothen']

        target_beamsize2 = np.sqrt(target_beamsize**2 - assumed_screen_res**2)
        sim_beamsize = screen_sim.gaussfit.sigma
        emittance = self.n_emittances[0]
        emittance_fit = emittance * (target_beamsize2 / sim_beamsize)**2
        return emittance_fit

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

    def matrix_forward(self, beamProfile, gaps, beam_offsets, manipulate_beam=None):
        """
        manipulate beam can be a function that takes the starting 6xN particle distribution, manipulates and returns it.
        """

        # wake_dict is needed in output for diagnostics
        wake_dict = {}

        def calc_wake(n_streaker, beam_before, tt_arr):
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
                wake_energy = np.interp(tt_arr, wake_tt, dipole_wake)
                dipole = wake_energy/self.energy_eV*np.sign(beamProfile.charge)
                if self.quad_wake:
                    quad_energy = np.interp(tt_arr, wake_tt, quad_wake)
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

        def calc_wake2d(n_streaker, beam_before, tt_arr):
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

            dict_dipole_2d = wf_model.wf2d(tt_arr*c, x_coords, gap/2., beamProfile.charge, wf_model.wxd, self.hist_bins_2d)
            dipole = dict_dipole_2d['wake_on_particles']/self.energy_eV

            if self.quad_wake:
                dict_quad_2d = wf_model.wf2d_quad(tt_arr*c, beam_before[0,:]+beam_offset, gap/2., beamProfile.charge, wf_model.wxq, self.hist_bins_2d)

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

        def split_streaker(n_streaker, beam_before, tt_arr):
            if self.split_streaker in (0, 1):
                beam_after = np.copy(beam_before)
                if self.wake2d:
                    dipole, quad = calc_wake2d(n_streaker, beam_before, tt_arr)
                else:
                    dipole, quad = calc_wake(n_streaker, beam_before, tt_arr)
                beam_after[1,:] += dipole + quad
                #print('Comb Dipole quad mean', np.abs(dipole+quad).mean())
            else:
                drift_minus = misc.drift(-self.struct_lengths[n_streaker]/2)[:4,:4]
                beam_after = beam_after = np.matmul(drift_minus, beam_before)
                delta_l = self.struct_lengths[n_streaker]/self.split_streaker/2
                drift = misc.drift(delta_l)[:4,:4]
                comb_effect = 0

                for n_split in range(self.split_streaker):
                    np.matmul(drift, beam_after, out=beam_after)

                    if self.wake2d:
                        dipole, quad = calc_wake2d(n_streaker, beam_after, tt_arr)
                    else:
                        dipole, quad = calc_wake(n_streaker, beam_after, tt_arr)
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

        streaker_matrices = self.simulator.get_streaker_matrices(self.timestamp, self.beamline)
        streaker_matrices2 = {}
        for key, arr in streaker_matrices.items():
            if key != 'mat_dict':
                streaker_matrices2[key] = arr[:4,:4]
        mat_dict = streaker_matrices['mat_dict']

        ## Generate beam

        # Optics
        if self.optics0 == 'default':
            optics_dict = config.default_optics[self.beamline]
            beta_x = optics_dict['beta_x']
            beta_y = optics_dict['beta_y']
            alpha_x = optics_dict['alpha_x']
            alpha_y = optics_dict['alpha_y']
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
        beam_start = np.array([watch['x'], watch['xp'], watch['y'], watch['yp']])

        if manipulate_beam is not None:
            beam_start = manipulate_beam(beam_start)

        ## Propagation of initial beam

        s1 = streaker_matrices2['start_to_s1']
        beam_before_s1 = np.matmul(s1, beam_start)
        beam_after_s1 = split_streaker(0, beam_before_s1, interp_tt)

        beam_before_s2 = np.matmul(streaker_matrices2['s1_to_s2'], beam_after_s1)
        beam_after_s2 = split_streaker(1, beam_before_s2, interp_tt)

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


        beam_at_screen4 = np.matmul(streaker_matrices2['s2_to_screen'], beam_after_s2)
        beam_at_screen = np.array([beam_at_screen4[0], beam_at_screen4[1], beam_at_screen4[2], beam_at_screen4[3], interp_tt, p_arr])

        beam0_at_screen4 = streaker_matrices2['s2_to_screen'] @ streaker_matrices2['s1_to_s2'] @ beam_before_s1
        beam0_at_screen = np.array([beam0_at_screen4[0], beam0_at_screen4[1], beam0_at_screen4[2], beam0_at_screen4[3], interp_tt, p_arr])
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


        screen = iap.getScreenDistributionFromPoints(beam_at_screen[0,:], self.screen_bins, 0, charge=beamProfile.charge)
        screen_no_smoothen = copy.deepcopy(screen)
        screen.smoothen(self.smoothen)

        for s_ in (screen_no_smoothen, screen):
            s_.reshape(self.len_screen)
            s_.cutoff(self.screen_cutoff)
            s_.normalize()

        r12_dict = {}
        screen_name = config.beamline_screens[self.beamline].replace('-','.')
        if self.beamline == 'Aramis':
            n_streakers = (0, 1)
        elif self.beamline == 'Athos':
            n_streakers = (0,)
        for n_streaker in n_streakers:
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict[screen_name]

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
                'beam_start': beam_start,
                'wake_dict': wake_dict,
                'bs_at_streaker': [beam_before_s1[0,:].std(), beam_before_s2[0,:].std()],
                'streaker_matrices': streaker_matrices,
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

        sim, mat_dict, wf_dicts, self.disp_dict = self.simulator.simulate_streaker(tt[mask], cc[mask], self.timestamp, 'file', None, self.energy_eV, wf_files=filenames, n_particles=self.n_particles, n_emittances=self.n_emittances, optics0=self.optics0)

        r12_dict = {}
        for n_streaker in (0, 1):
            r0 = mat_dict['MIDDLE_STREAKER_%i' % (n_streaker+1)]
            r1 = mat_dict['SARBD02.DSCR050']

            rr = np.matmul(r1, np.linalg.inv(r0))
            r12_dict[n_streaker] = rr[0,1]

        screen_watcher = sim.watch[-1]
        screen = iap.getScreenDistributionFromPoints(screen_watcher['x'], self.screen_bins, charge=beamProfile.charge)
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
                'disp_dict': self.disp_dict,
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
        test_profile = iap.get_gaussian_profile(40e-15, 200e-15, self.len_screen, 200e-12, self.energy_eV)
        forward_dict = self.matrix_forward(test_profile, [10e-3, 10e-3], [0, 0])
        self.bs_at_streaker = forward_dict['bs_at_streaker']


    def track_backward(self, screen, wake_effect, n_streaker, plot_details=False):
        screen = copy.deepcopy(screen)
        wake_time = wake_effect['t']
        wake_x = wake_effect['x']
        q_wake_x = wake_effect['quad']
        charge = wake_effect['charge']

        diff_x = np.diff(wake_x)
        if np.all(diff_x <= 0):
            wake_x = wake_x[::-1]
            q_wake_x = q_wake_x[::-1]
            wake_time = wake_time[::-1]
        elif np.all(diff_x >= 0):
            pass
        else:
            raise ValueError('Wake x is not monotonous')

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

        if self.quad_wake_back:
            if self.override_quad_beamsize:
                bs_at_streaker = self.quad_x_beamsize[n_streaker]
            else:
                if self.bs_at_streaker is None:
                    self.set_bs_at_streaker()
                bs_at_streaker = self.bs_at_streaker[n_streaker]

            screen.reshape(self.n_particles)
            rand0 = np.random.randn(len(screen.x))
            np.clip(rand0, -3, 3, out=rand0)
            randx = rand0*bs_at_streaker
            t_interp0 = np.zeros_like(screen.x)

            for n_x, (x, rx) in enumerate(zip(screen.x, randx)):
                t_interp0[n_x] = np.interp(x, wake_x+rx*q_wake_x, wake_time)

            charge_interp, hist_edges = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
            charge_interp[0] = 0
            charge_interp[-1] = 0
            t_interp = (hist_edges[1:] + hist_edges[:-1])/2.
        else:
            screen.reshape(self.n_particles)
            t_interp0 = np.interp(screen.x, wake_x, wake_time)
            charge_interp, hist_edges = np.histogram(t_interp0, bins=self.n_particles//100, weights=screen.intensity, density=True)
            charge_interp[0] = 0
            charge_interp[-1] = 0
            t_interp = (hist_edges[1:] + hist_edges[:-1])/2.

        try:
            if np.any(np.diff(t_interp) < 0):
                t_interp = t_interp[::-1]
                charge_interp = charge_interp[::-1]
            assert np.all(np.diff(t_interp) >= 0)
            bp = BeamProfile(t_interp, charge_interp, self.energy_eV, charge)
        except (ValueError, AssertionError) as e:
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
            raise
        bp.reshape(self.len_screen)
        bp.cutoff2(self.profile_cutoff)
        bp.crop()
        if np.any(np.isnan(bp.time)) or np.any(np.isnan(bp.current)):
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
        if beam_offsets[n_streaker] == 0:
            raise ValueError('Beam Offset is 0')
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

    def find_best_offset(self, offset0, offset_explore, tt_halfrange, meas_screen, gaps, profile, n_streaker, charge, prec=1e-6, method='centroid'):
        beam_offset_list = []
        sim_screens = []
        rms_list = []
        mean_list = []

        meas_screen = copy.deepcopy(meas_screen)
        meas_screen.reshape(self.len_screen)
        meas_screen.cutoff(self.screen_cutoff)
        meas_screen.crop()
        meas_screen.reshape(self.len_screen)
        centroid_meas = meas_screen.mean()
        rms_meas = meas_screen.rms()

        def forward(beam_offset):
            beam_offset = np.round(beam_offset/prec)*prec
            if beam_offset in beam_offset_list:
                return

            beam_offsets = [0., 0.]
            beam_offsets[n_streaker] = beam_offset

            sim_screen = self.matrix_forward(profile, gaps, beam_offsets)['screen']

            index = bisect.bisect(beam_offset_list, beam_offset)
            beam_offset_list.insert(index, beam_offset)
            sim_screens.insert(index, sim_screen)
            mean_list.insert(index, sim_screen.mean())
            rms_list.insert(index, sim_screen.rms())

        def get_index_min(output='index'):
            beam_offset_arr = np.array(beam_offset_list)
            if method == 'centroid':
                centroid_sim = np.array(mean_list)
                index_min = np.argmin(np.abs(centroid_sim - centroid_meas))
                sort = np.argsort(centroid_sim)
                beam_offset = np.interp(centroid_meas, centroid_sim[sort], beam_offset_arr[sort])
            elif method == 'rms' or method == 'beamsize':
                rms_sim = np.array(rms_list)
                index_min = np.argmin(np.abs(rms_sim - rms_meas))
                sort = np.argsort(rms_sim)
                beam_offset = np.interp(rms_meas, rms_sim[sort], beam_offset_arr[sort])
            else:
                raise ValueError('Method %s unknown' % method)

            if output == 'index':
                return index_min.squeeze()
            elif output == 'offset':
                return beam_offset

        beam_offset_arr = np.linspace(offset0-offset_explore, offset0+offset_explore, 3)
        for beam_offset in beam_offset_arr:
            forward(beam_offset)
        for _ in range(3):
            beam_offset = get_index_min(output='offset')
            forward(beam_offset)
        index = get_index_min(output='index')
        beam_offset = beam_offset_list[index]
        delta_offset = beam_offset - offset0

        output = {
                'sim_screens': sim_screens,
                'meas_screen': meas_screen,
                'sim_screen': sim_screens[index],
                'beam_offset': beam_offset,
                'beam_offsets': np.array(beam_offset_list),
                'delta_offset': delta_offset,
                'n_streaker': n_streaker,
                'gaps': gaps,
                'beam_offset0': offset0,
                'rms_arr': np.array(rms_list),
                'mean_arr': np.array(mean_list),
                }
        return output

    def find_best_gauss2(self, sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True, details=True, method='least_squares', delta_gap=(0., 0.), prec=None):

        if prec is None:
            prec = self.gauss_prec
        opt_func_values = []
        opt_func_screens = []
        opt_func_screens_no_smoothen = []
        opt_func_profiles = []
        opt_func_sigmas = []
        gauss_profiles = []
        gauss_wakes = []
        sig_t_list = []
        gaps = [gaps[0]+delta_gap[0], gaps[1]+delta_gap[1]]
        #distance = gaps[n_streaker]/2. - abs(beam_offsets[n_streaker])
        #print('gaps, beam_offsets, distance', gaps[n_streaker], beam_offsets[n_streaker], '%i' % (distance*1e6))

        #meas_screen = copy.deepcopy(meas_screen)
        meas_screen.reshape(self.len_screen)
        meas_screen.cutoff(self.screen_cutoff)
        meas_screen.crop()
        meas_screen.reshape(self.len_screen)
        centroid_meas = meas_screen.mean()
        rms_meas = meas_screen.rms()


        def gaussian_baf(sig_t):
            sig_t = np.round(sig_t/prec)*prec
            if sig_t in sig_t_list:
                return
            assert 1e-15 < sig_t < 1e-12

            bp_gauss = iap.get_gaussian_profile(sig_t, float(tt_halfrange), int(self.len_screen), float(charge), float(self.energy_eV))

            if self_consistent:
                bp_back0 = self.track_backward2(meas_screen, bp_gauss, gaps, beam_offsets, n_streaker)
            else:
                bp_back0 = bp_gauss

            baf = self.back_and_forward(meas_screen, bp_back0, gaps, beam_offsets, n_streaker)
            screen = copy.deepcopy(baf['screen'])
            diff = screen.compare(meas_screen)
            bp_out = baf['beam_profile']

            index = bisect.bisect(sig_t_list, sig_t)
            sig_t_list.insert(index, sig_t)
            opt_func_values.insert(index, (float(sig_t), diff))
            opt_func_screens.insert(index, screen)
            opt_func_profiles.insert(index, bp_out)
            opt_func_sigmas.insert(index, sig_t)
            opt_func_screens_no_smoothen.insert(index, baf['screen_no_smoothen'])
            gauss_profiles.insert(index, bp_gauss)
            gauss_wakes.insert(index, baf['wake_dict'])

        def get_index_min(output='index'):
            sig_t_arr = np.array(sig_t_list)
            if method == 'centroid':
                centroid_sim = np.array([x.mean() for x in opt_func_screens])
                index_min = np.argmin(np.abs(centroid_sim - centroid_meas))
                sort = np.argsort(centroid_sim)
                t_min = np.interp(centroid_meas, centroid_sim[sort], sig_t_arr[sort])
            elif method == 'rms' or method == 'beamsize':
                rms_sim = np.array([x.rms() for x in opt_func_screens])
                index_min = np.argmin(np.abs(rms_sim - rms_meas))
                sort = np.argsort(rms_sim)
                t_min = np.interp(rms_meas, rms_sim[sort], sig_t_arr[sort])
            else:
                raise ValueError('Method %s unknown' % method)

            if output == 'index':
                return index_min.squeeze()
            elif output == 't_sig':
                return t_min

        sig_t_arr = np.exp(np.linspace(np.log(sig_t_range.min()), np.log(sig_t_range.max()), 3))
        for sig_t in sig_t_arr:
            gaussian_baf(sig_t)

        for _ in range(3):
            sig_t_min = get_index_min(output='t_sig')
            gaussian_baf(sig_t_min)
        #sig_t_range2 = np.linspace(sig_t_min-2e-15, sig_t_min+2e-15, 5)
        #for sig_t in sig_t_range2:
        #    gaussian_baf(float(sig_t))
        index_min = get_index_min()
        if index_min == 0:
            print('Warning! index at left border!')
        if index_min == len(sig_t_list)-1:
            print('Warning! index at right border!')

        opt_func_values = np.array(opt_func_values)
        opt_value = opt_func_values[index_min, 1]
        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_sig_t = sig_t_list[index_min]
        best_gauss = gauss_profiles[index_min]
        best_wake = gauss_wakes[index_min]

        output = {
               'gauss_sigma': best_sig_t,
               'opt_value': opt_value,
               'reconstructed_screen': best_screen,
               'reconstructed_screen_no_smoothen': opt_func_screens_no_smoothen[index_min],
               'reconstructed_profile': best_profile,
               'best_gauss': best_gauss,
               'best_gauss_wake': gauss_wakes[index_min],
               ## Removed because ambiguous names
               #'final_screen': final_screen,
               #'final_profile': final_profile,
               'final_wake': best_wake,
               'meas_screen': meas_screen,
               'gaps': np.array(gaps),
               'beam_offsets': np.array(beam_offsets),
               'best_index': index_min,
               }
        # Final step
        if not self_consistent:
            baf = self.back_and_forward(meas_screen, best_profile, gaps, beam_offsets, n_streaker)
            final_screen = baf['screen']
            final_profile = baf['beam_profile']
            final_wake = baf['wake_dict']
            output['final_self_consistent_screen'] = final_screen
            output['final_self_consistent_profile'] = final_profile
            output['final_self_consistent_wake'] = final_wake


        if details:
            output.update({
                   'opt_func_values': opt_func_values,
                   'opt_func_screens': opt_func_screens,
                   'opt_func_profiles': opt_func_profiles,
                   'opt_func_sigmas': np.array(opt_func_sigmas),
                   'opt_func_wakes': gauss_wakes,
                   })

        return output


    def find_best_gauss_old(self, sig_t_range, tt_halfrange, meas_screen, gaps, beam_offsets, n_streaker, charge, self_consistent=True, details=True, method='least_squares'):

        opt_func_values = []
        opt_func_screens = []
        opt_func_screens_no_smoothen = []
        opt_func_profiles = []
        opt_func_sigmas = []
        gauss_profiles = []
        gauss_wakes = []
        sig_t_list = []

        #meas_screen = copy.deepcopy(meas_screen)
        meas_screen.reshape(self.len_screen)
        meas_screen.cutoff(self.screen_cutoff)
        meas_screen.crop()
        meas_screen.reshape(self.len_screen)

        def gaussian_baf(sig_t):
            if sig_t in sig_t_list:
                return
            assert 1e-15 < sig_t < 1e-12

            bp_gauss = iap.get_gaussian_profile(sig_t, float(tt_halfrange), int(self.len_screen), float(charge), float(self.energy_eV))

            if self_consistent:
                bp_back0 = self.track_backward2(meas_screen, bp_gauss, gaps, beam_offsets, n_streaker)
            else:
                bp_back0 = bp_gauss

            baf = self.back_and_forward(meas_screen, bp_back0, gaps, beam_offsets, n_streaker)
            screen = copy.deepcopy(baf['screen'])
            diff = screen.compare(meas_screen)
            bp_out = baf['beam_profile']

            index = bisect.bisect(sig_t_list, sig_t)
            sig_t_list.insert(index, sig_t)
            opt_func_values.insert(index, (float(sig_t), diff))
            opt_func_screens.insert(index, screen)
            opt_func_profiles.insert(index, bp_out)
            opt_func_sigmas.insert(index, sig_t)
            opt_func_screens_no_smoothen.insert(index, baf['screen_no_smoothen'])
            gauss_profiles.insert(index, bp_gauss)
            gauss_wakes.insert(index, baf['wake_dict'])

        def get_index_min():
            if method == 'least_squares':
                opt_func_values2 = np.array(opt_func_values)
                index_min = np.argmin(opt_func_values2[:, 1])
            elif method == 'centroid':
                centroid_meas = meas_screen.mean()
                centroid_sim = np.array([x.mean() for x in opt_func_screens])
                index_min = np.argmin(np.abs(centroid_sim - centroid_meas))
            elif method == 'rms' or method == 'beamsize':
                rms_meas = meas_screen.rms()
                rms_sim = np.array([x.rms() for x in opt_func_screens])
                index_min = np.argmin(np.abs(rms_sim - rms_meas))
            return index_min.squeeze()

        for sig_t in sig_t_range:
            gaussian_baf(float(sig_t))

        index_min = get_index_min()
        sig_t_min = sig_t_list[index_min]
        sig_t_range2 = np.linspace(sig_t_min-2e-15, sig_t_min+2e-15, 5)
        for sig_t in sig_t_range2:
            gaussian_baf(float(sig_t))
        index_min = get_index_min()
        if index_min == 0:
            print('Warning! index at left border!')
        if index_min == len(sig_t_list)-1:
            print('Warning! index at right border!')

        opt_func_values = np.array(opt_func_values)
        opt_value = opt_func_values[index_min, 1]
        best_screen = opt_func_screens[index_min]
        best_profile = opt_func_profiles[index_min]
        best_sig_t = sig_t_list[index_min]
        best_gauss = gauss_profiles[index_min]
        best_wake = gauss_wakes[index_min]

        output = {
               'gauss_sigma': best_sig_t,
               'opt_value': opt_value,
               'reconstructed_screen': best_screen,
               'reconstructed_screen_no_smoothen': opt_func_screens_no_smoothen[index_min],
               'reconstructed_profile': best_profile,
               'best_gauss': best_gauss,
               'best_gauss_wake': gauss_wakes[index_min],
               ## Removed because ambiguous names
               #'final_screen': final_screen,
               #'final_profile': final_profile,
               'final_wake': best_wake,
               'meas_screen': meas_screen,
               'gaps': np.array(gaps),
               'beam_offsets': np.array(beam_offsets),
               }
        # Final step
        if not self_consistent:
            baf = self.back_and_forward(meas_screen, best_profile, gaps, beam_offsets, n_streaker)
            final_screen = baf['screen']
            final_profile = baf['beam_profile']
            final_wake = baf['wake_dict']
            output['final_self_consistent_screen'] = final_screen
            output['final_self_consistent_profile'] = final_profile
            output['final_self_consistent_wake'] = final_wake


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

