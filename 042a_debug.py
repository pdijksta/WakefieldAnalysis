import matplotlib.pyplot as plt
import elegant_matrix
import pickle
import data_loader

plt.close('all')

elegant_matrix.set_tmp_dir('~/tmp_elegant/')

with open('/tmp/rec_args.pkl', 'rb') as f:
    kwargs, analysis_obj = pickle.load(f)

#screen_data = data_loader.

meas_screen = kwargs['meas_screen']
meas_screen.cutoff(10e-2)

analysis_obj.current_profile_rec_gauss(kwargs, True, None)


plt.show()

