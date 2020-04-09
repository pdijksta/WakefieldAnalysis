import functools
import numpy as np; np

#from ElegantWrapper.simulation import ElegantSimulation
from ElegantWrapper.watcher import FileViewer

#from wf_model import get_matrix_drift, get_matrix_quad, matmul
import data_loader
import elegant_matrix

storage_path = '/storage/'
# storage_path = '/mnt/usb/work/'

bpm_index = 1
streaker_index = 0

streakers = ['SARUN18.UDCP010', 'SARUN18.UDCP020']
bpms = ['SARBD02.DBPM040', 'SARBD02.DBPM010']
streaker = streakers[streaker_index]
bpm = bpms[bpm_index]

mag_file = storage_path + 'data_2020-02-03/quad_bnd.csv'
quad_file = storage_path + 'data_2020-02-03/sarun_18_19_quads.csv'
quad_file2 = storage_path + 'data_2020-02-03/sarbd01_quad_k1l.csv'

default_SF_par = FileViewer('./default.par.h5')


mag_data = data_loader.DataLoader(mag_file)
mag_data.add_other_csv(quad_file)
mag_data.add_other_csv(quad_file2)


@functools.lru_cache()
def get_magnet_length(mag_name):
    d = default_SF_par
    for name, par, value in zip(d['ElementName'], d['ElementParameter'], d['ParameterValue']):
        if name == mag_name and par == 'L':
            return value
        elif name == mag_name+'.Q1' and par == 'L':
            return value*2
    else:
        raise KeyError(mag_name)

def get_data(mag_name, timestamp):
    new_key = mag_name.replace('.','-')+':K1L-SET'
    return mag_data.get_prev_datapoint(new_key, timestamp)


quads = ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030']
tt = list(mag_data.values())[0][-2,0]

macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
for quad in quads:
    key = '_'+quad.lower()+'.k1_'
    val = get_data(quad, tt)
    length = get_magnet_length(quad)
    macro_dict[key] = val/length

cmd, sim = elegant_matrix.run_sim(macro_dict, './SwissFEL_in0.ele', './Elegant-Aramis-Reference.lat')

devices = quads + streakers + bpms + [q+'.COR' for q in quads]  + [q+'.Q2' for q in quads] + ['DRIFT1427'] + [s+'_B' for s in streakers]
mat_dict = {}
mat = sim.mat
for name, r11, r12, r21, r22 in zip(
        mat['ElementName'],
        mat['R11'],
        mat['R12'],
        mat['R21'],
        mat['R22'],
        ):
    if name in devices:
        mat_dict[name] = np.array([[r11, r12], [r21, r22]])

mat_elegant_new = mat_dict[bpm]

mat_elegant_new2 = elegant_matrix.get_elegant_matrix(0, tt)[bpm]

