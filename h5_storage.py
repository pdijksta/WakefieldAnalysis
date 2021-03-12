import getpass
import time
import re
from functools import lru_cache
import h5py
import numpy as np
import tracking

dt = h5py.special_dtype(vlen=bytes)

def stringDataset(group, name, data, system=None):
    dset = group.create_dataset(name, (1,), dtype=dt, data=data)
    if system:
        addSystemAttribute(dset, system)
    return dset

def addStringAttribute(dset_or_group, name, data):
    #return dset_or_group.attrs.create(name, np.string_(data)) # , (1,), dtype=dt)
    dset_or_group.attrs[name] = bytes(data, 'utf-8')

def addSystemAttribute(dset_or_group, data):
    return addStringAttribute(dset_or_group, 'system', data)

def add_dataset(group, name, data, system=None, dtype=None):
    if type(data) is str:
        return stringDataset(group, name, data, system)
    else:
        if dtype:
            dset = group.create_dataset(name, data=data, dtype=dtype)
        else:
            dset = group.create_dataset(name, data=data)
        if system:
            addSystemAttribute(dset, system)
        return dset

def saveH5Recursive(h5_filename, data_dict):

    def recurse_save(group, dict_or_data, dict_or_data_name, new_group=None):
        if dict_or_data is None:
            dict_or_data = 'None'
        if group is None:
            print("'recurse_save' has been called with None")
            raise ValueError
        if type(dict_or_data) is dict:

            try:
                new_group = group.create_group(dict_or_data_name)
            except:
                print(dict_or_data_name, 'error')
                #raise
            if new_group is None:
                raise ValueError
            for key, val in dict_or_data.items():
                try:
                    recurse_save(new_group, val, key)
                except ValueError:
                    print('I called recurse_save with None')
                    #import pdb; pdb.set_trace()
        else:
            mydata = dict_or_data
            inner_key = dict_or_data_name
            if type(mydata) is str:
                add_dataset(group, inner_key, mydata.encode('utf-8'), 'unknown')
            elif type(mydata) is tracking.BeamProfile:
                #raise NotImplementedError
                pass
            elif type(mydata) is tracking.ScreenDistribution:
                #raise NotImplementedError
                pass
            elif (type(mydata) is list and type(mydata[0]) is str) or (hasattr(mydata, 'dtype') and mydata.dtype.type is np.str_):
                # For list of strings, we need this procedure
                try:
                    if hasattr(mydata, 'dtype') and mydata.dtype.type is np.str and len(mydata.shape) == 2:
                        mydata = mydata.flatten()
                    if len(mydata.shape) == 2:
                        new_list = [[n.encode('ascii') for n in arr] for arr in mydata]
                        max_str_size = max(max(len(n) for n in arr) for arr in mydata)
                    elif len(mydata.shape) == 1:
                        new_list = [n.encode('ascii') for n in mydata]
                        max_str_size = max(len(n) for n in mydata)
                    elif len(mydata.shape) == 0:
                        new_list = [mydata.encode('ascii')]
                        max_str_size = len(new_list[0])
                    #print('Max len %i' % max_str_size)
                    dset = group.create_dataset(inner_key, mydata.shape, 'S%i' % max_str_size, new_list)
                    #print(np.array(dset))
                    dset.attrs.create('system', 'unknown', (1,), dtype=dt)

                except:
                    print('Error', inner_key)
                    print(type(mydata))
                    if type(mydata) is list:
                        print('type(mydata[0])')
                        print(type(mydata[0]))
                    print('mydata')
                    print(mydata)

            elif hasattr(mydata, 'dtype') and mydata.dtype == np.dtype('O'):
                if mydata.shape == ():
                    add_dataset(group, inner_key, mydata, 'unknown')
                elif len(mydata.shape) == 1:
                    add_dataset(group, inner_key, mydata, 'unknown')
                else:
                    for i in range(mydata.shape[0]):
                        for j in range(mydata.shape[1]):
                            try:
                                add_dataset(group, inner_key+'_%i_%i' % (i,j), mydata[i,j], 'unknown')
                            except:
                                print('Error')
                                print(group, inner_key, i, j)
            else:
                try:
                    add_dataset(group, inner_key, mydata, 'unknown')
                except Exception as e:
                    print('Error', e)
                    print(inner_key, type(mydata))

    with h5py.File(h5_filename, 'w') as dataH5:
        for main_key, subdict in data_dict.items():
            recurse_save(dataH5, subdict, main_key, None)
        #recurse_save(dataH5, data_dict, 'none', new_group=dataH5)

def loadH5Recursive(h5_file):
    def recurse_load(group_or_val, key, saved_dict_curr):
        type_ = type(group_or_val)

        try:
            group_or_val[()]
        except:
            hasval = False
        else:
            hasval = True

        if type_ is h5py._hl.files.File:
            for new_key, new_group_or_val in group_or_val.items():
                recurse_load(new_group_or_val, new_key, saved_dict_curr)
        elif type_ is h5py._hl.group.Group:
            saved_dict_curr[key] = new_dict = {}
            for new_key, new_group_or_val in group_or_val.items():
                recurse_load(new_group_or_val, new_key, new_dict)
        elif type_ == np.dtype('O') and hasval and type(group_or_val[()]) is bytes:
            saved_dict_curr[key] = group_or_val[()].decode()
        elif type_ == h5py._hl.dataset.Dataset:
            dtype = group_or_val.dtype
            #if not hasattr(group_or_val, 'value'):
            #    print('Could not store key %s with type %s in dict' % (key, dtype))
            #    return
            if dtype in (np.dtype('int64'), np.dtype('int32'), np.dtype('int16'), np.dtype('int8'), np.dtype('uint32'), np.dtype('uint16'), np.dtype('uint8')):
                saved_dict_curr[key] = np.array(group_or_val[()], int).squeeze()
            elif dtype == np.dtype('bool'):
                try:
                    saved_dict_curr[key] = bool(group_or_val[()])
                except:
                    print('Could not store key %s with type %s in dict (1)' % (key, dtype))
            elif dtype in (np.dtype('float64'), np.dtype('float32'), np.dtype('uint16'), np.dtype('uint64')):
                saved_dict_curr[key] = np.array(group_or_val[()]).squeeze()
            elif dtype.str.startswith('|S'):
                if group_or_val[()].shape == (1,1):
                    saved_dict_curr[key] = group_or_val[()][0,0].decode()
                elif group_or_val[()].shape == (1,):
                    saved_dict_curr[key] = group_or_val[()][0].decode()

                elif group_or_val[()].shape == ():
                    saved_dict_curr[key] = group_or_val[()].decode()
                else:
                    saved_dict_curr[key] = [x.decode() for x in group_or_val[()].squeeze()]
            elif dtype.str == '|O':
                saved_dict_curr[key] = group_or_val[()]
            elif type(group_or_val[()]) is str:
                saved_dict_curr[key] = group_or_val[()]
            else:
                print('Could not store key %s with type %s in dict (2)' % (key, dtype))
        else:
            print('Could not store key %s with type %s in dict (3)' % (key, type_))

    saved_dict = {}
    with h5py.File(h5_file, 'r') as f:
        if 'none' in f:
            recurse_load(f['none'], 'key', saved_dict)
            saved_dict = saved_dict['key']
        else:
            recurse_load(f, 'key', saved_dict)
    return saved_dict

def save_h5_new(saved_dict, h5_file):

    def recurse_save(dict_, group, system):
        print('recurse', dict_.keys())
        for key, subdict_or_data in dict_.items():
            type_ = type(subdict_or_data)
            print(key, type_)
            if type_ is dict:
                new_group = group.create_group(key)
                recurse_save(subdict_or_data, new_group, system)
            elif type_ is np.ndarray:
                add_dataset(group, key, subdict_or_data, system)
            elif type_ is str:
                add_dataset(group, key, subdict_or_data, system, dtype=dt)
            else:
                raise ValueError(key, type_)

    @lru_cache()
    def re_axis(x):
        return re.compile('gr_%s_axis_(\d+)_(\d+)' % x)

    @lru_cache()
    def re_gauss_function(x):
        return re.compile('gr_%s_fit_gauss_function_(\d+)_(\d+)' % x)

    n_measurements, n_images = saved_dict['Raw_data']['image'].shape[:2]

    # Create arrays for gr / slice values, that differ in size for different n_measurements, n_images
    gr_x_shape_max = -1
    gr_y_shape_max = -1
    for key, data in sorted(saved_dict['Raw_data'].items()):
        if key.startswith('gr_x_axis'):
            gr_x_shape_max = max(gr_x_shape_max, data.shape[0])
        elif key.startswith('gr_y_axis'):
            gr_y_shape_max = max(gr_y_shape_max, data.shape[0])

    gr_x_axis = np.zeros([n_measurements, n_images, gr_x_shape_max])*np.nan
    gr_y_axis = np.zeros([n_measurements, n_images, gr_y_shape_max])*np.nan
    gr_x_fit_gauss_function = gr_x_axis.copy()
    gr_y_fit_gauss_function = gr_y_axis.copy()

    for key, data in sorted(saved_dict['Raw_data'].items()):
        for arr, regex in [
                (gr_x_axis, re_axis('x')),
                (gr_y_axis, re_axis('y')),
                (gr_x_fit_gauss_function, re_gauss_function('x')),
                (gr_y_fit_gauss_function, re_gauss_function('y')),
                ]:
            match = regex.match(key)
            if match is not None:
                #print(key, 'matches', regex)
                n_measurement, n_image = map(int, match.groups())
                arr[n_measurement, n_image,:len(data)] = data
                continue

    with h5py.File(h5_file, 'w') as f:
        general = f.create_group('general')
        stringDataset(general, 'user', getpass.getuser())
        stringDataset(general, 'application', 'EmittanceTool')
        stringDataset(general, 'author', 'Philipp Dijkstal and Eduard Prat')
        stringDataset(general, 'created', time.ctime())

        experiment = f.create_group('experiment')
        try:
            from epics import caget
            lrr = float(caget('SIN-TIMAST-TMA:Beam-Exp-Freq-RB'))
        except Exception as e:
            print('Could not obtain Laser rep rate!')
            print(e)
            lrr = np.nan
        add_dataset(experiment, 'Laser rep rate', lrr, 'unknown')
        # TBD: save snapshot here

        scan1 = f.create_group('scan 1')

        method = scan1.create_group('method')
        method.create_dataset('records', data=[float(n_measurements)])
        method.create_dataset('samples', data=[float(n_images)])
        method.create_dataset('dimension', data=[1])
        stringDataset(method, 'type', 'Line scan')
        recurse_save(saved_dict['Input'], method, 'Application Input')


        data = scan1.create_group('data')

        screen = data.create_group(saved_dict['Input']['Profile monitor'])
        recurse_save(saved_dict['Meta_data'], screen, 'Emittance data')


        for key, data_ in sorted(saved_dict['Raw_data'].items()):
            if not any([x.match(key) for x in [re_axis('x'), re_axis('y'), re_gauss_function('x'), re_gauss_function('y')]]):
                add_dataset(screen, key, data_, 'Camera')
                #print('Created %s' % key)

        if not np.all(np.isnan(gr_x_axis)):
            add_dataset(screen, 'gr_x_axis', gr_x_axis, 'Camera')
        else:
            print('gr_x_axis is nan')
        if not np.all(np.isnan(gr_y_axis)):
            add_dataset(screen, 'gr_y_axis', gr_y_axis, 'Camera')
        else:
            print('gr_y_axis is nan')
        if not np.all(np.isnan(gr_x_fit_gauss_function)):
            add_dataset(screen, 'gr_x_fit_gauss_function', gr_x_fit_gauss_function, 'Camera')
        else:
            print('gr_x_fit_gauss_function is nan')
        if not np.all(np.isnan(gr_y_fit_gauss_function)):
            add_dataset(screen, 'gr_y_fit_gauss_function', gr_y_fit_gauss_function, 'Camera')
        else:
            print('gr_y_fit_gauss_function is nan')

        if 'Magnet_data' in saved_dict:
            for n_magnet, magnet in enumerate(saved_dict['Magnet_data']['Magnets']):
                mag_group = method.create_group('actuators/%s' % magnet)
                add_dataset(mag_group, 'K', saved_dict['Magnet_data']['K'][n_magnet], 'Magnet')
                add_dataset(mag_group, 'I-SET', saved_dict['Magnet_data']['I-SET'][n_magnet], 'Magnet')
        elif not saved_dict['Input']['Dry run'] in (np.array(False), False):
            raise ValueError('No magnet data')
        else:
            print('Magnet data not saved.')

