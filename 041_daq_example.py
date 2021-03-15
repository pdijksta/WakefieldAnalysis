import h5py
import numpy as np
import sys; sys
import argparse

import logging
import pyscan
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address

parser = argparse.ArgumentParser()
parser.add_argument('n_images', help='number of images to take', type=int)
parser.add_argument('h5_filename', type=str, help='Name of file where results are saved (h5 format)')
parser.add_argument('--dry-run', action='store_true', help='Use simulation screen')
args = parser.parse_args()

n_intervals = args.n_images
#time_interval = max(1/caget('SIN-TIMAST-TMA:Beam-Appl-Freq-RB'), 0.3)
time_interval = 1.1
print(time_interval, n_intervals)

time_positioner = pyscan.TimePositioner(time_interval=time_interval, n_intervals=n_intervals)
readables = [
        'bs://gr_x_fit_standard_deviation',
        'bs://gr_y_fit_standard_deviation',
        'bs://gr_x_fit_mean',
        'bs://gr_y_fit_mean',
        'bs://gr_x_axis',
        'bs://gr_y_axis',
        'bs://gr_x_fit_gauss_function',
        'bs://gr_y_fit_gauss_function',
        'bs://image',
        'bs://x_axis',
        'bs://y_axis',
        ]

settings = pyscan.scan_settings(settling_time=0.01,
                         measurement_interval=0.2, # questionable
                         n_measurements=1)

if args.dry_run:
    screen = 'simulation'
else:
    screen = 'SARBD01.DSCR110'

pipeline_client = PipelineClient("http://sf-daqsync-01:8889/")
cam_instance_name = str(screen) + "_sp1"
stream_address = pipeline_client.get_instance_stream(cam_instance_name)
stream_host, stream_port = get_host_port_from_stream_address(stream_address)

# Configure bsread
pyscan.config.bs_default_host = stream_host
pyscan.config.bs_default_port = stream_port

logging.getLogger("mflow.mflow").setLevel(logging.ERROR)


raw_output = pyscan.scan(positioner=time_positioner, readables=readables, settings=settings)
output = [[x] for x in raw_output]

def pyscan_result_to_dict(readables, result, scrap_bs=False):
    """
    Excpects a nested list of order 3.
    Level 1 is the scan index.
    Level 2 is the number of images per scan index (unless this number is 1 in which case this level does not exist).
    Level 3 is the number of readables.

    Returns a shuffled version that takes the form of the dictionary, with the readables as keys.
    """

    output = {}

    for nR, readable in enumerate(readables):
        readable_output1 = []
        for level_scan in result:
            readable_output2 = []
            for level_image in level_scan:
                readable_output2.append(level_image[nR])
            readable_output1.append(readable_output2)

        if scrap_bs and hasattr(readable, 'startswith') and readable.startswith('bs://'):
            readable2 = readable[5:]
        else:
            readable2 = readable

        try:
            output[readable2] = np.array(readable_output1)
        except:
            output[readable2] = readable_output1

    return output

output_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)

def saveH5Recursive(h5_filename, Emit_data):

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

    def recurse_save(group, dict_or_data, dict_or_data_name, new_group=None):
        if group is None:
            print("'recurse_save' has been called with None")
            raise ValueError
        if type(dict_or_data) is dict:
            new_group = group.create_group(dict_or_data_name)
            if new_group is None:
                import pdb; pdb.set_trace()
            for key, val in dict_or_data.items():
                try:
                    recurse_save(new_group, val, key)
                except ValueError:
                    print('I called recurse_save with None')
                    import pdb; pdb.set_trace()
        else:
            mydata = dict_or_data
            inner_key = dict_or_data_name
            if type(mydata) is str:
                try:
                    add_dataset(group, inner_key, mydata.encode('utf-8'), 'unknown')
                except:
                    import pdb; pdb.set_trace()
            elif (type(mydata) is list and type(mydata[0]) is str) or (hasattr(mydata, 'dtype') and mydata.dtype.type is np.str_):
                # For list of strings, we need this procedure
                try:
                    if hasattr(mydata, 'dtype') and mydata.dtype.type is np.str and len(mydata.shape) == 2:
                        mydata = mydata.flatten()
                    new_list = [n.encode('ascii') for n in mydata]
                    max_str_size = max(len(n) for n in mydata)
                    #print('Max len %i' % max_str_size)
                    dset = group.create_dataset(inner_key, (len(new_list), 1), 'S%i' % max_str_size, new_list)
                    #print(np.array(dset))
                    dset.attrs.create('system', 'unknown', (1,), dtype=dt)

                except:
                    print('Error', inner_key)
                    print(type(mydata))
                    if type(mydata) is list:
                        print(type(mydata[0]))
                    print(mydata)

            elif hasattr(mydata, 'dtype') and mydata.dtype == np.dtype('O'):
                if mydata.shape == ():
                    add_dataset(group, inner_key, mydata, 'unknown')
                elif len(mydata.shape) == 1:
                    add_dataset(group, inner_key, mydata, 'unknown')
                    #import pdb; pdb.set_trace()
                    #raise ValueError('This only works for 2-dim arrays. %s is: %i-dim array' % (inner_key, len(mydata.shape)))
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
                    print(e)
                    print(inner_key, type(mydata))


    with h5py.File(h5_filename, 'w') as dataH5:
        recurse_save(dataH5, Emit_data, 'none', new_group=dataH5)



saveH5Recursive(args.h5_filename, output_dict)

