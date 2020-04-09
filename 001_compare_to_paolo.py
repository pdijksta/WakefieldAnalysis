from scipy.constants import c
from wf_model import wxd
from h5_storage import loadH5Recursive
from scipy.io import loadmat

saved = loadH5Recursive('./example_current_profile.h5')
paolos_saved = loadmat('./wxd.mat')
paolos_wxd = paolos_saved['wxd']
paolos_s = paolos_saved['s'].squeeze()

time_profile = saved['time_profile']
current = saved['current']
energy_eV = saved['energy_eV']


s = time_profile*c
s -= s.min()

a = 2.5e-3/2

my_wxd = wxd(paolos_s, a)


