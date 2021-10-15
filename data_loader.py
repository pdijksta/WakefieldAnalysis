import functools
import json
import numpy as np
import h5py
from scipy.io import loadmat

try:
    from h5_storage import loadH5Recursive
except ImportError:
    from .h5_storage import loadH5Recursive

@functools.lru_cache(5)
def csv_to_dict(file_):
    data = np.loadtxt(file_, str, delimiter=';')
    assert len(data[0]) % 3 == 0

    var_list = [str(key).split('.')[0] for key in data[0,::3]]
    dict_ = {var: [] for var in var_list}

    for row in data[1:]:
        for n_var, var in enumerate(var_list):
            base_index = n_var*3
            ms, t, d = np.array([0,1,2]) + base_index

            empty = (row[t] in ('""', ''))
            if not empty:
                data_row = [float(row[ms]), float(row[d])]
                dict_[var].append(data_row)

    for key, value in dict_.items():
        dict_[key] = np.array(value, float)

    return dict_

def json_to_dict(file_):
    with open(file_) as f:
        abc = json.load(f)
    outp = {}
    for subdict in abc:
        data = subdict['data']
        data_list = []
        for dd in data:
            data_list.append([float(dd['globalSeconds']), dd['value']])
        outp[subdict['channel']['name']] = np.array(data_list)
    return outp

class DataLoader(dict):
    def __init__(self, file_csv=None, file_json=None, file_h5=None):
        if file_csv is not None:
            dict_ = csv_to_dict(file_csv)
            super().__init__(dict_)
        if file_json is not None:
            self.add_other_json(file_json)
        if file_h5 is not None:
            self.add_other_h5(file_h5)

    def get_prev_datapoint(self, key, timestamp, verbose=False):
        data_list = self[key]
        for index, (timestamp2, _) in enumerate(data_list):
            if timestamp2 > timestamp:
                out_index = index-1
                if out_index in (-1, len(data_list-1)):
                    #import pdb; pdb.set_trace()
                    raise KeyError('Requested data at border of data array. Key %s' % key)
                if verbose:
                    print(key, out_index)
                return data_list[index-1,1]
        else:
            #return data_list[-1,1]
            #import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
            raise KeyError('Requested data at border of data array')

    def add_other_csv(self, file_):
        new_dict = csv_to_dict(file_)
        assert len(set(new_dict.keys()).intersection(set(self.keys()))) == 0
        self.update(new_dict)

    def add_other_json(self, file_):
        new_dict = json_to_dict(file_)
        assert len(set(new_dict.keys()).intersection(set(self.keys()))) == 0
        self.update(new_dict)

    def add_other_h5(self, file_):
        new_dict = loadH5Recursive(file_)
        assert len(set(new_dict.keys()).intersection(set(self.keys()))) == 0
        self.update(new_dict)

@functools.lru_cache(5)
def load_blmeas(file_):
    with h5py.File(file_, 'r') as f:
        if 'Ambient data' in f:
            return load_blmeas_new(f, file_)
        else:
            return load_blmeas_old(f, file_)

def load_blmeas_old(f, file_):

    energy_eV = np.array(f['Input']['Energy']).squeeze()*1e6
    output = {'energy_eV': energy_eV}

    # Not always are both zero crossings measured
    suffixes = [('', '1')]
    if 'Time axes 2' in f['Meta_data']:
        suffixes.append((' 2', '2'))

    for suffix, suffix_out in suffixes:
        time_profile = np.array(f['Meta_data']['Time axes'+suffix])*1e-15
        current = np.array(f['Meta_data']['Current profile'+suffix])

        # Reorder if time array is descending
        if time_profile[1] - time_profile[0] < 0:
            time_profile = time_profile[::-1]
            current = current[::-1]

        # Find out where is the head and the tail.
        # In our conventios, the head is at negative time
        centroids = np.array(f['Raw_data']['Beam centroids'+suffix])
        phases = np.arange(0, len(centroids)) # phases always ascending, and we only care about the sign of the fit

        if len(phases) > 1:
            dy_dphase = np.polyfit(phases, centroids.mean(axis=1), 1)[0]

            sign_dy_dt = np.sign(dy_dphase)
        else:
            print('Warning! Time orientation of bunch length measurement cannot be determined!')
            print(file_)
            sign_dy_dt = 1

        if sign_dy_dt == -1:
            current = current[::-1]

        output.update({
            'time_profile'+suffix_out: time_profile,
            'current'+suffix_out: current,
        })

    return output

def load_blmeas_new(f, file_):
    energy_eV = np.array(f['Input data']['Energy']).squeeze()*1e6
    output = {'energy_eV': energy_eV}

    # Not always are both zero crossings measured
    suffixes = [('', '1')]
    if 'Time axis 2' in f['Processed data']:
        suffixes.append((' 2', '2'))

    for suffix, suffix_out in suffixes:
        time_profile = np.array(f['Processed data']['Time axis'+suffix])*1e-15
        current = np.array(f['Processed data']['Current profile'+suffix])

        ## Reorder if time array is descending
        #if time_profile[1] - time_profile[0] < 0:
        #    time_profile = time_profile[::-1]
        #    current = current[::-1]

        ## Find out where is the head and the tail.
        ## In our conventios, the head is at negative time
        #centroids = np.array(f['Raw_data']['Beam centroids'+suffix])
        #phases = np.arange(0, len(centroids)) # phases always ascending, and we only care about the sign of the fit

        #if len(phases) > 1:
        #    dy_dphase = np.polyfit(phases, centroids.mean(axis=1), 1)[0]

        #    sign_dy_dt = np.sign(dy_dphase)
        #else:
        #    print('Warning! Time orientation of bunch length measurement cannot be determined!')
        #    print(file_)
        #    sign_dy_dt = 1

        #if sign_dy_dt == -1:
        #    current = current[::-1]

        output.update({
            'time_profile'+suffix_out: time_profile,
            'current'+suffix_out: current,
        })

    return output


# For application

def load_screen_data(filename_or_dict, key, index):
    if type(filename_or_dict) is dict:
        dict_ = filename_or_dict
    elif filename_or_dict.endswith('.h5'):
        dict_ = loadH5Recursive(filename_or_dict)
    elif filename_or_dict.endswith('.mat'):
        dict_ = loadmat(filename_or_dict)
    else:
        raise ValueError('Must be h5 or mat file. Is: %s' % filename_or_dict)

    dict0 = dict_
    if 'pyscan_result' in dict0:
        dict_ = dict_['pyscan_result']
        key = 'image'
    else:
        dict_ = dict0

    if 'x_axis_m' not in dict_:
        print(dict_.keys())
    x_axis = dict_['x_axis_m']
    data = dict_[key].astype(float)
    if index not in ('None', None):
        index = int(index)
        data = data[index]

    if len(data.shape) == 2:
        # Assume saved data are already projections
        projx = data
    elif len(data.shape) == 3:
        # Assume saved data are images
        projx = np.zeros((data.shape[0], len(x_axis)))
        for n_img, img in enumerate(data):
            try:
                projx[n_img,:] = img.sum(axis=0)
            except:
                import pdb; pdb.set_trace()
    else:
        raise ValueError('Expect shape of 2 or 3. Is: %i' % len(data.shape))

    y_axis = dict_['y_axis_m'] if 'y_axis_m' in dict_ else None

    if np.abs(x_axis.max()) > 1:
        x_axis *= 1e-6
        print('Converting x_axis from um to m')
    if np.abs(y_axis.max()) > 1:
        y_axis *= 1e-6
        print('Converting y_axis from um to m')

    # TBD
    # - Add y information. Needed for lasing reconstruction.

    if x_axis[1] < x_axis[0]:
        x_axis = x_axis[::-1]
        projx = projx[:,::-1]
    if y_axis[1] < y_axis[0]:
        y_axis = y_axis[::-1]

    output = {
            'x_axis': x_axis,
            'projx': projx,
            'y_axis': y_axis,
            }
    if 'meta_data' in dict0:
        output['meta_data'] = dict0['meta_data']
    if 'meta_data_end' in dict0:
        output['meta_data'] = dict0['meta_data_end']
    return output

