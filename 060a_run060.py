import os

#for method in ('rms', 'centroid', 'least_squares'):
for method in ('centroid', 'rms'):
    for file_index in (2, 3):
        for offset_index in (0,):
            os.system('python3 ./060_self_consistent_gap.py %i %i %s --noshow' % (file_index, offset_index, method))

