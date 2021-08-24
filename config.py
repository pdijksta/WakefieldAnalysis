import numpy as np
import itertools
import collections

streaker_names = {
        'Aramis': collections.OrderedDict([
            (0, 'SARUN18-UDCP010'),
            (1, 'SARUN18-UDCP020'),
            ]),
        'Athos': collections.OrderedDict([
            (0, 'SATMA02-UDCP045'),
            ]),
        }

streaker_lengths = {
        'SARUN18-UDCP010': 1.,
        'SARUN18-UDCP020': 1.,
        'SATMA02-UDCP045': 1.,
        }

beamline_quads = {
        'Aramis': ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030'],
        'Athos': ['SATMA02.MQUA050', 'SATBD01.MQUA010', 'SATBD01.MQUA030', 'SATBD01.MQUA050', 'SATBD01.MQUA070', 'SATBD01.MQUA090', 'SATBD02.MQUA030'],
        }

beamline_chargepv = {
        'Aramis': 'SINEG01-DICT215:B1_CHARGE-OP',
        'Athos': 'SINEG01-DICT215:B2_CHARGE-OP',
        }

beamline_energypv = {
        'Aramis': 'SARBD01-MBND100:ENERGY-OP',
        'Athos': 'SATBD01-MBND200:ENERGY-OP',
        }

beamline_screens = {
        'Aramis': 'SARBD02-DSCR050',
        'Athos': 'SATBD02-DSCR050',
        }

gas_monitor_pvs = {
        'Aramis': 'SARFE10-PBPG050:PHOTON-ENERGY-PER-PULSE-AVG',
        }

default_optics = {
        'Aramis': {
            'beta_x': 4.968,
            'alpha_x': -0.563,
            'beta_y': 16.807,
            'alpha_y': 1.782,
            },
        'Athos': {
            'beta_x': 30.9,
            'alpha_x': 3.8,
            'beta_y': 69.4,
            'alpha_y': -14.3,
            },
        }

_aramis_pvs = ['SARUN%02i-DBPM070:%s1' % (i, dim) for i, dim in itertools.product(range(1, 21), ('X', 'Y'))]
_aramis_pvs += ['SARBD01-DBPM040:%s1' % dim for dim in ('X', 'Y')]
_aramis_pvs += ['SARBD02-DBPM010:%s1' % dim for dim in ('X', 'Y')]

beamline_bpm_pvs = {
        'Aramis': _aramis_pvs,
        }

beamline_charge_pvs_bsread = {
        'Aramis': ['SARBD01-DICT030:B1_CHARGE', 'SINEG01-DICT215:B1_CHARGE'],
        }

all_streakers = []
for beamline, beamline_dict in streaker_names.items():
    all_streakers.extend([x for x in beamline_dict.values()])

get_default_tracker_settings = lambda: {
        'magnet_file': None,
        'timestamp': None,
        'struct_lengths': np.array([1., 1.]),
        'n_particles': int(100e3),
        'n_emittances': np.array([755, 755])*1e-9,
        'screen_bins': 500,
        'screen_cutoff': 2e-2,
        'smoothen': 20e-6,
        'profile_cutoff': 1e-2,
        'len_screen': 2000,
        'quad_wake': False,
        'bp_smoothen': 1e-15,
        'override_quad_beamsize': False,
        'quad_x_beamsize': np.array([10., 10.])*1e-6,
        }

get_default_gauss_recon_settings = lambda: {
        'self_consistent': True,
        'sig_t_range': np.exp(np.linspace(np.log(7), np.log(100), 15))*1e-15,
        'tt_halfrange': 200e-15,
        'charge': 200e-12,
        'method': 'centroid',
        'delta_gap': (0., 0.)
        }

tmp_elegant_dir = '~/tmp_elegant'

fontsize = 8

rho_label = r'$\rho$ (nC/m)'

