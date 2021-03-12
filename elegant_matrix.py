import functools
import itertools
import glob
import os
import shutil
import datetime
import numpy as np
from scipy.constants import c

from ElegantWrapper.simulation import ElegantSimulation
from ElegantWrapper.watcher import FileViewer, Watcher2

try:
    import data_loader
    import wf_model
except ImportError:
    from . import data_loader
    from . import wf_model

pid = os.getpid()
ctr = 0

streakers = ['SARUN18.UDCP010', 'SARUN18.UDCP020']

this_dir = os.path.abspath(os.path.dirname(__file__))

default_SF_par = FileViewer(os.path.join(this_dir, './default.par.h5'))
default_SF_par_athos = FileViewer(os.path.join(this_dir, './default_athos.par.h5'))
symlink_files = glob.glob(os.path.join(this_dir, './elegant_wakes/wake*.sdds'))

quads = ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030']
quads_athos = ['SATMA02.MQUA050', 'SATBD01.MQUA010', 'SATBD01.MQUA030', 'SATBD01.MQUA050', 'SATBD01.MQUA070', 'SATBD01.MQUA090', 'SATBD02.MQUA030']


#tmp_dir = '/home/philipp/tmp_elegant/'
tmp_dir = None
mute_elegant = True

def set_tmp_dir(_tmp_dir):
    global tmp_dir
    tmp_dir = os.path.expanduser(_tmp_dir)

def clear_tmp_dir():
    dirs = os.path.listdir(tmp_dir)
    for dir_ in dirs:
        path = tmp_dir+'/'+dir_
        shutil.rmtree(path)
        print('Removed %s' % path)

def get_timestamp(year, month, day, hour, minute, second):
    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    timestamp = int(date.strftime('%s'))
    return timestamp

def run_sim(macro_dict, ele, lat, copy_files=(), move_files=(), symlink_files=(), del_sim=True):
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

    now = datetime.datetime.now()
    if tmp_dir is None:
        raise NameError('Error: Set tmp_dir first!')

    new_dir = os.path.join(tmp_dir, 'elegant_%i_%i-%02i-%02i_%02i-%02i-%02i_%i' % (pid, now.year, now.month, now.day, now.hour, now.minute, now.second, ctr))
    os.makedirs(new_dir)
    for f in copy_files:
        shutil.copy(f, new_dir)
    for f in move_files:
        shutil.move(f, new_dir)
    for f in symlink_files:
        os.symlink(f, os.path.join(new_dir, os.path.basename(f)))
    ctr += 1
    new_ele_file = shutil.copy(ele, new_dir)
    shutil.copy(lat, new_dir)
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        cmd = 'elegant %s ' % os.path.basename(new_ele_file)
        for key, val in macro_dict.items():
            cmd += ' -macro=%s=%s' % (key, val)
        if mute_elegant:
            cmd += ' >/dev/null'
        #print(cmd)
        with open(os.path.join(new_dir, 'run.sh'), 'w') as f:
            f.write(cmd+'\n')
        status = os.system(cmd)
        if status != 0:
            raise RuntimeError('Elegant failed! %s' % new_dir)
    finally:
        os.chdir(old_dir)

    sim = ElegantSimulation(new_ele_file, del_sim=del_sim)
    return cmd, sim

@functools.lru_cache(5)
def gen_beam(nemitx, nemity, alphax, betax, alphay, betay, p_central, rms_bunch_duration, n_particles):
    print('Gen beam  called with emittances %.1e %.1e' % (nemitx, nemity))
    macro_dict = {
            '_nemitx_': nemitx,
            '_nemity_': nemity,
            '_alphax_': alphax,
            '_betax_': betax,
            '_alphay_': alphay,
            '_betay_': betay,
            '_p_central_': p_central,
            '_bunch_length_': rms_bunch_duration*c,
            '_n_particles_': n_particles,
            }
    lat = os.path.join(this_dir, './gen_beam.lat')
    ele = os.path.join(this_dir, './gen_beam.ele')

    cmd, sim = run_sim(macro_dict, ele, lat)
    w = sim.watch[-1]
    return w, sim

def get_magnet_length(mag_name, branch='Aramis'):
    if branch == 'Aramis':
        d = default_SF_par
    elif branch == 'Athos':
        d = default_SF_par_athos
    for name, par, value in zip(d['ElementName'], d['ElementParameter'], d['ParameterValue']):
        if name == mag_name and par == 'L':
            return value
        elif name == mag_name+'.Q1' and par == 'L':
            return value*2
    else:
        #import pdb; pdb.set_trace()
        raise KeyError(mag_name)


class simulator:

    beta_x0 = 4.968
    alpha_x0 = -0.563
    beta_y0 = 16.807
    alpha_y0 = 1.782

    def __init__(self, file_):
        file_h5, file_json = None, None
        if '.json' in file_:
            file_json = file_
        elif '.h5' in file_:
            file_h5 = file_
        else:
            raise ValueError(file_)
        self.mag_data = data_loader.DataLoader(file_json=file_json, file_h5=file_h5)

    def get_data_quad(self, mag_name, timestamp):
        new_key = mag_name.replace('.','-')+':K1L-SET'
        return self.get_data(new_key, timestamp)

    def get_data(self, key, timestamp):
        return self.mag_data.get_prev_datapoint(key, timestamp)

    @functools.lru_cache()
    def get_elegant_matrix(self, streaker_index, timestamp, del_sim=True, print_=False, branch='Aramis'):
        """
        streaker_index must be in (0,1)
        """

        if branch == 'Aramis':
            lat = os.path.join(this_dir, './Elegant-Aramis-Reference.lat')
            ele = os.path.join(this_dir, './SwissFEL_in0.ele')

            if streaker_index != 'NULL':
                macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
            else:
                macro_dict = {'_matrix_start_': 'Q'}

            for quad in quads:
                key = '_'+quad.lower()+'.k1_'
                val = self.get_data_quad(quad, timestamp)
                length = get_magnet_length(quad)
                k1 = val/length
                macro_dict[key] = k1
                if print_:
                    print(key, '%.2e' % k1)

        elif branch == 'Athos':
            lat = os.path.join(this_dir, './Athos_Full.lat')
            ele = os.path.join(this_dir, './SwissFEL_in0_Athos.ele')
            macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER$1'}

            for quad in quads_athos:
                key = '_'+quad.lower()+'.k1_'
                val = self.get_data_quad(quad, timestamp)
                length = get_magnet_length(quad, branch='Athos')
                k1 = val/length
                macro_dict[key] = k1
                if print_:
                    print(key, '%.2e' % k1)


        cmd, sim = run_sim(macro_dict, ele, lat, symlink_files=symlink_files)
        sim.del_sim = del_sim

        mat_dict = {}
        disp_dict = {}
        mat = sim.mat
        #try:
        #    mat['ElementName']
        #except:
        #    import pdb; pdb.set_trace()
        for name, r11, r12, r21, r22, r36 in zip(
                mat['ElementName'],
                mat['R11'],
                mat['R12'],
                mat['R21'],
                mat['R22'],
                mat['R36'],
                ):
            #mat_dict[name] = np.array([[r11, r12], [r21, r22]])
            disp_dict[name] = r36

        for index, name in enumerate(mat['ElementName']):
            mat_dict[name] = a = np.zeros([6,6])
            for i,j in itertools.product(range(6), repeat=2):
                a[i,j] = mat['R%i%i' % (i+1,j+1)][index]

        return mat_dict, disp_dict

    def get_streaker_matrices(self, timestamp, del_sim=True):

        mat_dict = self.get_elegant_matrix('NULL', timestamp, del_sim=del_sim)[0]
        s1 = mat_dict['MIDDLE_STREAKER_1']
        s2 = mat_dict['MIDDLE_STREAKER_2']
        screen = mat_dict['SARBD02.DSCR050']
        s1_to_s2 = np.matmul(s2, np.linalg.inv(s1))
        s2_to_screen = np.matmul(screen, np.linalg.inv(s2))

        # s2_to_screen @ s1_to_s2 @ s1 == screen -> checked

        return {
                'start_to_s1': s1,
                's1_to_s2': s1_to_s2,
                's2_to_screen': s2_to_screen,
                'mat_dict': mat_dict,
                }

    def simulate_streaker(self, current_time, current_profile, timestamp, gaps, beam_offsets, energy_eV, del_sim=True, n_particles=int(20e3), linearize_twf=True, wf_files=None, charge=200e-12, n_emittances=(300e-9, 300e-9), optics0='default'):
        """
        gaps can be 'file', then wf_files must be specified. Else, wf_files is ignored.
        Returns: sim, mat_dict, wf_dicts, disp_dict
        """

        n_particles = int(n_particles)
        curr = current_profile
        tt = current_time
        integrated_curr = np.cumsum(curr)
        integrated_curr /= integrated_curr[-1]

        randoms = np.random.rand(n_particles)
        interp_tt = np.interp(randoms, integrated_curr, tt)

        p_central = energy_eV/511e3

        #watcher0 = Watcher(os.path.join(os.path.dirname(__file__), 'SwissFEL0-001.w1.h5'))
        #new_watcher_dict = {'t': interp_tt}
        #for key in ('p', 'x', 'y', 'xp', 'yp'):
        #    arr = watcher0[key]
        #    xx = np.linspace(arr.min(), arr.max(), 1000.)
        #    hist, bin_edges = np.histogram(arr, bins=xx)
        #    arr_cum = np.cumsum(hist).astype(float)
        #    arr_cum /= arr_cum.max()
        #    randoms = np.random.rand(n_particles)
        #    interp = np.interp(randoms, arr_cum, xx[:-1]+np.diff(xx)[0])
        #    new_watcher_dict[key] = interp

        #from Sven's OpticsServer (new version), 06.04.2020
        #location: SARUN18.START
        #&twiss_output
        #        filename	= %s.twi,
        #        matched		= 0,
        #        beta_x = 	4.968
        #        alpha_x =  -0.563
        #        beta_y =    16.807
        #        alpha_y =   1.782
        #&end

        # Used before:
        #beta_x = 5.067067
        #beta_y = 16.72606
        #alpha_x = -0.5774133
        #alpha_y = 1.781136

        if optics0 == 'default':
            beta_x, alpha_x, beta_y, alpha_y = self.beta_x0, self.alpha_x0, self.beta_y0, self.alpha_y0
        else:
            beta_x, alpha_x, beta_y, alpha_y = optics0


        watcher0, sim0 = gen_beam(n_emittances[0], n_emittances[1], alpha_x, beta_x, alpha_y, beta_y, p_central, 20e-6/c, n_particles)
        #import pdb; pdb.set_trace()

        new_watcher_dict = {'t': interp_tt}
        for key in ('p', 'x', 'y', 'xp', 'yp'):
            new_watcher_dict[key] = watcher0[key].copy()
        del watcher0
        sim0.__del__()

        new_watcher = Watcher2({}, new_watcher_dict)
        new_watcher_file = '/tmp/input_beam.sdds'
        new_watcher.toSDDS(new_watcher_file)

        max_xx = (interp_tt.max() - interp_tt.min())*c*1.1
        xx = np.linspace(0, max_xx, int(1e4))

        filenames = ('/tmp/streaker1.sdds', '/tmp/streaker2.sdds')
        wf_dicts = []
        if gaps == 'file':
            shutil.copy(wf_files[0], filenames[0])
            shutil.copy(wf_files[1], filenames[1])
        else:
            for gap, beam_offset, filename in zip(gaps, beam_offsets, filenames):
                wf_dict = wf_model.generate_elegant_wf(filename, xx, gap/2., beam_offset, L=1.)
                wf_dicts.append(wf_dict)

        lat = os.path.join(this_dir, './Aramis.lat')
        ele = os.path.join(this_dir, './SwissFEL_in_streaker.ele')
        macro_dict = {
                '_p_central_': p_central,
                '_twf_factor_': int(linearize_twf),
                '_charge_': '%e' % charge,
                }
        for quad in quads:
            key = '_'+quad.lower()+'.k1_'
            val = self.get_data_quad(quad, timestamp)
            length = get_magnet_length(quad)
            k1 = val/length
            macro_dict[key] = k1

        move_files = (new_watcher_file,) + filenames
        cmd, sim = run_sim(macro_dict, ele, lat, move_files=move_files, symlink_files=symlink_files)
        sim.del_sim = del_sim

        mat_dict = {}
        disp_dict = {}
        mat = sim.mat

        for name, r11, r12, r21, r22, r16 in zip(
                mat['ElementName'],
                mat['R11'],
                mat['R12'],
                mat['R21'],
                mat['R22'],
                mat['R16'],
                ):
            mat_dict[name] = np.array([[r11, r12], [r21, r22]])
            disp_dict[name] = r16

        return sim, mat_dict, wf_dicts, disp_dict

#@functools.wraps(simulator)
@functools.lru_cache(400)
def get_simulator(file_):
    return simulator(file_)

