import itertools

streaker_names = {
        'Aramis': {
            0: 'SARUN18-UDCP010',
            1: 'SARUN18-UDCP020',
            },
        }

beamline_quads = {
        'Aramis': ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030'],
        'Athos': ['SATMA02.MQUA050', 'SATBD01.MQUA010', 'SATBD01.MQUA030', 'SATBD01.MQUA050', 'SATBD01.MQUA070', 'SATBD01.MQUA090', 'SATBD02.MQUA030'],
        }

beamline_chargepv = {
        'Aramis': 'SINEG01-DICT215:B1_CHARGE-OP',
        'Athos': 'SINEG01-DICT215:B2_CHARGE-OP',
        }

gas_monitor_pvs = {
        'Aramis': 'SARFE10-PBPG050:PHOTON-ENERGY-PER-PULSE-AVG',
        }

_aramis_pvs = ['SARUN%02i-DBPM070:%s1' % (i, dim) for i, dim in itertools.product(range(1, 21), ('X', 'Y'))]
_aramis_pvs += ['SARBD01-DBPM040:%s1' % dim for dim in ('X', 'Y')]
_aramis_pvs += ['SARBD02-DBPM010:%s1' % dim for dim in ('X', 'Y')]

beamline_bpm_pvs = {
        'Aramis': _aramis_pvs,
        }

all_streakers = []
for beamline, beamline_dict in streaker_names.items():
    all_streakers.extend([x for x in beamline_dict.values()])
