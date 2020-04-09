import numpy as np; np

from ElegantWrapper.simulation import ElegantSimulation

from wf_model import get_matrix_drift, get_matrix_quad, matmul
import data_loader
import elegant_matrix

storage_path = '/storage/'
# storage_path = '/mnt/usb/work/'

override = True
bpm_index = 1
streaker_index = 0

mag_file = storage_path + 'data_2020-02-03/quad_bnd.csv'
quad_file = storage_path + 'data_2020-02-03/sarun_18_19_quads.csv'
quad_file2 = storage_path + 'data_2020-02-03/sarbd01_quad_k1l.csv'


mag_data = data_loader.DataLoader(mag_file)
mag_data.add_other_csv(quad_file)
mag_data.add_other_csv(quad_file2)

def get_data(mag_name, timestamp):
    if 'SARBD02' not in mag_name:
        length = 0.08
    else:
        length = 0.3

    if override:
        par = sim.par
        if 'SARBD02' not in mag_name:
            mag_name += '.Q1'
        for name, par, val in zip(par['ElementName'], par['ElementParameter'], par['ParameterValue']):
            if name == mag_name and par == 'K1':
                return val * length, length
    else:
        new_key = mag_name.replace('.','-')+':K1L-SET'
        return mag_data.get_prev_datapoint(new_key, timestamp), length


sim = ElegantSimulation(storage_path + 'elegant_files/ec_delk_0.00/SwissFEL_in3.ele')


streaker1 = 'SARUN18.UDCP010'
streaker2 = 'SARUN18.UDCP020'
streakers = [streaker1, streaker2]

bpm1 = 'SARBD02.DBPM040'
bpm2 = 'SARBD02.DBPM010'
bpms = [bpm1, bpm2]

quads = ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030']
quad_positions = []
for quad in quads:
    try:
        quad_position = sim.get_element_position(quad+'.COR', mean=True)
    except KeyError:
        quad_position = sim.get_element_position(quad, mean=True)
    quad_positions.append(quad_position)


s_streaker1 = sim.get_element_position(streaker1, mean=True)
s_streaker2 = sim.get_element_position(streaker2, mean=True)

s_streaker = s_streaker1, s_streaker2

s_bpm1 = sim.get_element_position(bpm1, mean=True)
s_bpm2 = sim.get_element_position(bpm2, mean=True)

s_bpm = s_bpm1, s_bpm2


def get_static_matrix(timestamp):
    quad_matrices = []
    for quad in quads:
        k1l, length = get_data(quad, timestamp)
        quad_matrices.append(get_matrix_quad(k1l, length))

    all_matrices = [quad_matrices[0]]
    for n_q, quad in enumerate(quads[1:], 1):
        dist = quad_positions[n_q] - quad_positions[n_q-1]
        print(quad_positions[n_q], quad_positions[n_q-1], dist)
        all_matrices.append(get_matrix_drift(dist))
        all_matrices.append(quad_matrices[n_q])

    return matmul(all_matrices[::-1])

tt = list(mag_data.values())[0][-2,0]

def get_matrix(timestamp, bpm_index, streaker_index):
    d1 = get_matrix_drift(quad_positions[0] - s_streaker[streaker_index])
    d2 = get_matrix_drift(s_bpm[bpm_index] - quad_positions[-1])
    mat0 = get_static_matrix(timestamp)

    return matmul([d2, mat0, d1])

mat_analytic = get_matrix(tt, bpm_index, streaker_index)


mat_dict = {}

bpm = bpms[bpm_index]
streaker = streakers[streaker_index]

devices = quads + streakers + bpms + [q+'.COR' for q in quads]  + [q+'.Q2' for q in quads] + ['DRIFT1427'] + [s+'_B' for s in streakers]
if override:
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

    mat_elegant = mat_dict[bpm] @ np.linalg.inv(mat_dict[streaker])

macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
for quad in quads:
    key = '_'+quad.lower()+'.k1_'
    val, length = get_data(quad, tt)
    macro_dict[key] = val/length

cmd, sim_new = elegant_matrix.run_sim(macro_dict, './SwissFEL_in0.ele', './Elegant-Aramis-Reference.lat')
sim_new.del_sim = True

mat_dict_new = {}
mat = sim_new.mat
for name, r11, r12, r21, r22 in zip(
        mat['ElementName'],
        mat['R11'],
        mat['R12'],
        mat['R21'],
        mat['R22'],
        ):
    if name in devices:
        mat_dict_new[name] = np.array([[r11, r12], [r21, r22]])

mat_elegant_new = mat_dict_new[bpm]



#mat_elegant = mat_dict[quads[0]+'.Q2'] @ np.linalg.inv(mat_dict['DRIFT1427'])
#mat_analytic = matmul([
#    get_matrix_quad(get_data(quads[0], None)),
#    get_matrix_drift(quad_positions[0] - (s_streaker1-0.5))
#    ])
#
