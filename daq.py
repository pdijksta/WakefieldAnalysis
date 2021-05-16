#import itertools
import numpy as np
import logging
import datetime

import pyscan
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from epics import caget

import config

def get_readables(beamline):
    return [
            'bs://image',
            'bs://x_axis',
            'bs://y_axis',
            config.beamline_chargepv[beamline],
            ]


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

def get_images(screen, n_images, beamline='Aramis'):

    meta_dict_1 = get_meta_data()

    positioner = pyscan.BsreadPositioner(n_messages=n_images)
    settings = pyscan.scan_settings(settling_time=0.01, measurement_interval=0.2, n_measurements=1)

    pipeline_client = PipelineClient("http://sf-daqsync-01:8889/")
    cam_instance_name = str(screen) + "_sp1"
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    logging.getLogger("mflow.mflow").setLevel(logging.ERROR)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings)
    output = [[x] for x in raw_output]

    result_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)

    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6 # convert to m
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0,0,:]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0,:]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data()

    output_dict = {
            'pyscan_result': result_dict,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }

    return output_dict

def data_streaker_offset(streaker, offset_range, screen, n_images, dry_run, beamline='Aramis'):


    meta_dict_1 = get_meta_data()

    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    offset_pv = streaker+':CENTER'

    current_val = caget(offset_pv)

    # Start from closer edge of scan
    if abs(current_val - offset_range[0]) > abs(current_val - offset_range[-1]):
        offset_range = offset_range[::-1]

    if dry_run:
        screen = 'simulation'
        writables = None
        positioner = pyscan.TimePositioner(time_interval=(1.1*n_images), n_intervals=len(offset_range))

    else:
        positions = offset_range * 1e3 # convert to mm
        positioner = pyscan.VectorPositioner(positions=positions.tolist())
        writables = [pyscan.epics_pv(pv_name=offset_pv, readback_pv_name=offset_pv+'.RBV', tolerance=0.005)]

    cam_instance_name = screen + '_sp1'
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    logging.getLogger('mflow.mflow').setLevel(logging.ERROR)

    settings = pyscan.scan_settings(settling_time=1, n_measurements=n_images, write_timeout=60)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)
    result_dict = pyscan_result_to_dict(readables, raw_output, scrap_bs=True)
    #import pdb; pdb.set_trace()
    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0][0]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data()

    output = {
            'pyscan_result': result_dict,
            'streaker_offsets': offset_range,
            'screen': screen,
            'n_images': n_images,
            'dry_run': dry_run,
            'streaker': streaker,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }
    return output

def get_aramis_quad_strengths():
    quads = config.beamline_quads['Aramis']

    k1l_dict = {}
    for quad in quads:
        k1l_dict[quad] = caget(quad.replace('.', '-')+':K1L-SET')
    energy_pv = 'SARBD01-MBND100:ENERGY-OP'
    k1l_dict[energy_pv] = caget(energy_pv)
    return k1l_dict

def get_meta_data():
    all_streakers = config.all_streakers
    meta_dict = {}
    meta_dict.update({x+':GAP': caget(x+':GAP') for x in all_streakers})
    meta_dict.update({x+':CENTER': caget(x+':CENTER') for x in all_streakers})

    k1l_dict = get_aramis_quad_strengths()
    meta_dict.update(k1l_dict)
    meta_dict['time'] = str(datetime.datetime.now())
    for gas_monitor_energy_pv in config.gas_monitor_pvs.values():
        meta_dict[gas_monitor_energy_pv] = caget(gas_monitor_energy_pv)
    return meta_dict

