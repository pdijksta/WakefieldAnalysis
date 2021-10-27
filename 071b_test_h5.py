import numpy as np
import h5py

with h5py.File('/tmp/test.h5', 'w') as f:
    group = f.create_group('Test')
    group.create_dataset(name='abc', data=np.zeros(5))
    group.create_dataset(name='abc', data=np.zeros(6))
