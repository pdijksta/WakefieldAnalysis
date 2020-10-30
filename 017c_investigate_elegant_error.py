
import numpy as np

sim_dir = '/home/philipp/tmp_elegant/elegant_75920_2020-10-28_19-09-02_5/'

inp = np.loadtxt(sim_dir + 'input_beam.sdds', skiprows=10)
wake = np.loadtxt(sim_dir + 'streaker1.sdds', skiprows=9)

