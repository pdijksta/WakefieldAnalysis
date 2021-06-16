import os

for method in ('centroid', 'rms'):
    for file_index in (2, 3):
        os.system('python3 ./061_recon_rms_method.py --file_index %i --method %s --noshow' % (file_index, method))

