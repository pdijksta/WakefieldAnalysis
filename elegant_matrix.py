import functools
import os
import shutil
import datetime
import numpy as np

from ElegantWrapper.simulation import ElegantSimulation
from ElegantWrapper.watcher import FileViewer

import data_loader

pid = os.getpid()
ctr = 0

storage_path = '/storage/'
# storage_path = '/mnt/usb/work/'
streakers = ['SARUN18.UDCP010', 'SARUN18.UDCP020']

mag_file = storage_path + 'data_2020-02-03/quad_bnd.csv'
quad_file = storage_path + 'data_2020-02-03/sarun_18_19_quads.csv'
quad_file2 = storage_path + 'data_2020-02-03/sarbd01_quad_k1l.csv'

default_SF_par = FileViewer('./default.par.h5')


mag_data = data_loader.DataLoader(mag_file)
mag_data.add_other_csv(quad_file)
mag_data.add_other_csv(quad_file2)

quads = ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030']


def run_sim(macro_dict, ele, lat):
    """
    macro_dict must have  the following form:
    {'_matrix_start_': 'MIDDLE_STREAKER_1$1',
     '_sarun18.mqua080.k1_': 2.8821223408771512,
     '_sarun19.mqua080.k1_': -2.7781958823761546,
     '_sarun20.mqua080.k1_': 2.8206489094677942,
     '_sarbd01.mqua020.k1_': -2.046012575644645,
     '_sarbd02.mqua030.k1_': -0.7088921626501486}

     returns elegant command and elegant simulation object

    """
    global ctr
    old_dir = os.getcwd()

    now = datetime.datetime.now()
    new_dir = '/tmp/elegant_%i_%i-%02i-%02i_%02i-%02i-%02i_%i' % (pid, now.year, now.month, now.day, now.hour, now.minute, now.second, ctr)
    os.system('mkdir -p %s' % new_dir)
    ctr += 1
    new_ele_file = shutil.copy(ele, new_dir)
    shutil.copy(lat, new_dir)
    try:
        os.chdir(new_dir)
        cmd = 'elegant %s 1>/dev/null' % os.path.basename(new_ele_file)
        for key, val in macro_dict.items():
            cmd += ' -macro=%s=%s' % (key, val)
        #print(cmd)
        os.system(cmd)
    finally:
        os.chdir(old_dir)

    sim = ElegantSimulation(new_ele_file)
    return cmd, sim

@functools.lru_cache(400)
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

@functools.lru_cache()
def get_elegant_matrix(streaker_index, timestamp, del_sim=True, print_=False):
    """
    streaker_index must be in (0,1)
    """

    #streaker = streakers[streaker_index]
    #bpm = bpms[bpm_index]

    macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
    for quad in quads:
        key = '_'+quad.lower()+'.k1_'
        val = get_data(quad, timestamp)
        length = get_magnet_length(quad)
        k1 = val/length
        macro_dict[key] = k1
        if print_:
            print(key, '%.2e' % k1)

    cmd, sim = run_sim(macro_dict, './SwissFEL_in0.ele', './Elegant-Aramis-Reference.lat')
    sim.del_sim = del_sim

    #devices = quads + streakers + bpms + [q+'.COR' for q in quads] \
    #    + ['MIDDLE_STREAKER_%i' % (s+1) for s in range(len(streakers))]

    mat_dict = {}
    mat = sim.mat
    for name, r11, r12, r21, r22 in zip(
            mat['ElementName'],
            mat['R11'],
            mat['R12'],
            mat['R21'],
            mat['R22'],
            ):
        mat_dict[name] = np.array([[r11, r12], [r21, r22]])

    return mat_dict

if __name__ == '__main__':
    ele = './SwissFEL_in0.ele'
    lat = './Elegant-Aramis-Reference.lat'
    macro_dict = {
            '_sarun18.mqua080.k1_': 1,
            '_sarun19.mqua080.k1_': 1,
            '_sarun20.mqua080.k1_': 1,
            '_sarbd01.mqua020.k1_': 1,
            '_sarbd02.mqua030.k1_': 1,
            '_matrix_start_': 'MIDDLE_STREAKER_1$1',
            }

    sim, cmd = run_sim(macro_dict, ele, lat)
    sim.del_sim = True

