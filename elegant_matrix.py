import functools
import glob
import os
import shutil
import datetime
import numpy as np
from scipy.constants import c

from ElegantWrapper.simulation import ElegantSimulation
from ElegantWrapper.watcher import FileViewer, Watcher, Watcher2

import data_loader
import wf_model

pid = os.getpid()
ctr = 0

# storage_path = '/mnt/usb/work/'
streakers = ['SARUN18.UDCP010', 'SARUN18.UDCP020']

#storage_path = '/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/data_2020-02-03'
#mag_file = storage_path + 'data_2020-02-03/quad_bnd.csv'
#quad_file = storage_path + 'data_2020-02-03/sarun_18_19_quads.csv'
#quad_file2 = storage_path + 'data_2020-02-03/sarbd01_quad_k1l.csv'

default_SF_par = FileViewer('./default.par.h5')
default_SF_par_athos = FileViewer('./default_athos.par.h5')
symlink_files = glob.glob(os.path.join(os.path.abspath(os.path.dirname('.')), './elegant_wakes/wake*.sdds'))


#mag_data = data_loader.DataLoader(mag_file)
#mag_data.add_other_csv(quad_file)
#mag_data.add_other_csv(quad_file2)

#mag_data = data_loader.DataLoader(file_json='/afs/psi.ch/intranet/SF/Beamdynamics/Philipp/data/archiver_api_data/2020-08-26.json11')
quads = ['SARUN18.MQUA080', 'SARUN19.MQUA080', 'SARUN20.MQUA080', 'SARBD01.MQUA020', 'SARBD02.MQUA030']
quads_athos = ['SATMA02.MQUA050', 'SATBD01.MQUA010', 'SATBD01.MQUA030', 'SATBD01.MQUA050', 'SATBD01.MQUA070', 'SATBD01.MQUA090', 'SATBD02.MQUA030']

def get_timestamp(year, month, day, hour, minute, second):
    date = datetime.datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    timestamp = int(date.strftime('%s'))
    return timestamp

class simulator:
    def __init__(self, file_json):
        self.mag_data = data_loader.DataLoader(file_json=file_json)

    def run_sim(self, macro_dict, ele, lat, copy_files=(), move_files=(), symlink_files=()):
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
        for f in copy_files:
            shutil.copy(f, new_dir)
        for f in move_files:
            shutil.move(f, new_dir)
        for f in symlink_files:
            os.symlink(f, os.path.join(new_dir, os.path.basename(f)))
        ctr += 1
        new_ele_file = shutil.copy(ele, new_dir)
        shutil.copy(lat, new_dir)
        try:
            os.chdir(new_dir)
            cmd = 'elegant %s ' % os.path.basename(new_ele_file)
            for key, val in macro_dict.items():
                cmd += ' -macro=%s=%s' % (key, val)
            print(cmd)
            os.system(cmd)
        finally:
            os.chdir(old_dir)

        sim = ElegantSimulation(new_ele_file)
        return cmd, sim

    @functools.lru_cache(400)
    def get_magnet_length(self, mag_name, branch='Aramis'):
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

    def get_data(self, mag_name, timestamp):
        new_key = mag_name.replace('.','-')+':K1L-SET'
        return self.mag_data.get_prev_datapoint(new_key, timestamp)

    @functools.lru_cache()
    def get_elegant_matrix(self, streaker_index, timestamp, del_sim=True, print_=False, branch='Aramis'):
        """
        streaker_index must be in (0,1)
        """

        #streaker = streakers[streaker_index]
        #bpm = bpms[bpm_index]

        macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
        if branch == 'Aramis':
            lat = './Elegant-Aramis-Reference.lat'
            ele = './SwissFEL_in0.ele'
            macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER_%i$1' % (streaker_index+1)}
            for quad in quads:
                key = '_'+quad.lower()+'.k1_'
                val = self.get_data(quad, timestamp)
                length = self.get_magnet_length(quad)
                k1 = val/length
                macro_dict[key] = k1
                if print_:
                    print(key, '%.2e' % k1)

        elif branch == 'Athos':
            lat = './Athos_Full.lat'
            ele = './SwissFEL_in0_Athos.ele'
            macro_dict = {'_matrix_start_': 'MIDDLE_STREAKER$1'}

            for quad in quads_athos:
                key = '_'+quad.lower()+'.k1_'
                val = self.get_data(quad, timestamp)
                length = self.get_magnet_length(quad, branch='Athos')
                k1 = val/length
                macro_dict[key] = k1
                if print_:
                    print(key, '%.2e' % k1)


        cmd, sim = self.run_sim(macro_dict, ele, lat, symlink_files=symlink_files)
        sim.del_sim = del_sim

        mat_dict = {}
        mat = sim.mat
        #try:
        #    mat['ElementName']
        #except:
        #    import pdb; pdb.set_trace()
        for name, r11, r12, r21, r22 in zip(
                mat['ElementName'],
                mat['R11'],
                mat['R12'],
                mat['R21'],
                mat['R22'],
                ):
            mat_dict[name] = np.array([[r11, r12], [r21, r22]])

        return mat_dict

    def simulate_streaker(self, current_time, current_profile, timestamp, gaps, beam_offsets, energy_eV, del_sim=True, n_particles=int(20e3)):
        """
        Returns: sim, mat_dict
        """

        n_particles = int(n_particles)
        curr = current_profile
        tt = current_time
        integrated_curr = np.cumsum(curr)
        integrated_curr /= integrated_curr.max()

        randoms = np.random.rand(n_particles)
        interp_tt = np.interp(randoms, integrated_curr, tt)

        p_central = energy_eV/511e3

        watcher0 = Watcher(os.path.join(os.path.dirname(__file__), 'SwissFEL0-001.w1.h5'))
        new_watcher_dict = {'t': interp_tt}
        for key in ('p', 'x', 'y', 'xp', 'yp'):
            arr = watcher0[key]
            xx = np.linspace(arr.min(), arr.max(), 1000.)
            hist, bin_edges = np.histogram(arr, bins=xx)
            arr_cum = np.cumsum(hist).astype(float)
            arr_cum /= arr_cum.max()
            randoms = np.random.rand(n_particles)
            interp = np.interp(randoms, arr_cum, xx[:-1]+np.diff(xx)[0])
            new_watcher_dict[key] = interp

        pp = new_watcher_dict['p']
        pp = pp / pp.mean() * p_central
        new_watcher_dict['p'] = pp

        new_watcher = Watcher2({}, new_watcher_dict)
        new_watcher_file = '/tmp/input_beam.sdds'
        new_watcher.toSDDS(new_watcher_file)

        max_xx = (interp_tt.max() - interp_tt.min())*c*1.1
        xx = np.linspace(0, max_xx, int(1e4))

        filenames = ('/tmp/streaker1.sdds', '/tmp/streaker2.sdds')
        wf_dicts = []
        for gap, beam_offset, filename in zip(gaps, beam_offsets, filenames):
            wf_dict = wf_model.generate_elegant_wf(filename, xx, gap/2., beam_offset, L=1.)
            wf_dicts.append(wf_dict)

        lat = './Aramis.lat'
        ele = './SwissFEL_in_streaker.ele'
        macro_dict = {'_p_central_': p_central}
        for quad in quads:
            key = '_'+quad.lower()+'.k1_'
            val = self.get_data(quad, timestamp)
            length = self.get_magnet_length(quad)
            k1 = val/length
            macro_dict[key] = k1

        move_files = (new_watcher_file,) + filenames
        cmd, sim = self.run_sim(macro_dict, ele, lat, move_files=move_files, symlink_files=symlink_files)
        sim.del_sim = del_sim

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

        return sim, mat_dict, wf_dicts

#@functools.wraps(simulator)
@functools.lru_cache(400)
def get_simulator(file_json):
    return simulator(file_json)



