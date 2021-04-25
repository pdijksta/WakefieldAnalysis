
streaker_names = {
        'Aramis': {
            0: 'SARUN18-UDCP010',
            1: 'SARUN18-UDCP020',
            },
        }

gas_monitor_pvs = {
        'Aramis': 'SARFE10-PBPG050:PHOTON-ENERGY-PER-PULSE-AVG',
        }

all_streakers = []
for beamline, beamline_dict in streaker_names.items():
    all_streakers.extend([x for x in beamline_dict.values()])
