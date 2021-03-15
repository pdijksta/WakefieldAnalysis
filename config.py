
streaker_names = {
        'Aramis': {
            0: 'SARUN18-UDCP010',
            1: 'SARUN18-UDCP020',
            },
        }

all_streakers = []
for beamline, beamline_dict in streaker_names.items():
    all_streakers.extend([x for x in beamline_dict.values()])
