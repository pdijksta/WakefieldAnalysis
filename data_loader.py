import functools
import json
import numpy as np

from h5_storage import loadH5Recursive

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
    def __init__(self, file_csv=None, file_json=None):
        if file_csv is not None:
            dict_ = csv_to_dict(file_csv)
            super().__init__(dict_)
        if file_json is not None:
            self.add_other_json(file_json)

    def get_prev_datapoint(self, key, timestamp, verbose=True):
        data_list = self[key]
        for index, (timestamp2, _) in enumerate(data_list):
            if timestamp2 > timestamp:
                out_index = index-1
                if out_index in (-1, len(data_list-1)):
                    raise KeyError('Requested data at border of data array. Key %s' % key)
                if verbose:
                    print(key, out_index)
                return data_list[index-1,1]
        else:
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

@functools.lru_cache(5)
def load_blmeas(file_):
    bl_meas = loadH5Recursive(file_)
    time_profile1, current1 = bl_meas['Meta_data']['Time axes'][::-1]*1e-15, bl_meas['Meta_data']['Current profile'][::-1]
    try:
        time_profile2, current2 = bl_meas['Meta_data']['Time axes 2'][::-1]*1e-15, bl_meas['Meta_data']['Current profile 2'][::-1]
    except KeyError:
        time_profile2, current2 = None, None
    energy_eV = bl_meas['Input']['Energy'].squeeze()*1e6

    return {
            'time_profile1': time_profile1,
            'time_profile2': time_profile2,
            'current1': current1,
            'current2': current2,
            'energy_eV': energy_eV,
            'all_data': bl_meas
            }


if __name__ == '__main__':
    file_ = '/storage/data_2020-02-03/sarun_18_19_quads.csv'
    dict_ = csv_to_dict(file_)
    data_loader = DataLoader(file_)
    file_json = '/storage/data_2020-02-03/2020-02-03.json1'
    data_loader.add_other_jason(file_json)


