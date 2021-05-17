#import itertools
import time
import numpy as np
import logging
import datetime

import pyscan
from cam_server import PipelineClient, CamClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB
from epics import caget, caput

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

    print('Start get_images for screen %s, %i images, beamline %s' % (screen, n_images, beamline))

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

    print('End get_images')

    return output_dict

def data_streaker_offset(streaker, offset_range, screen, n_images, dry_run, beamline='Aramis'):

    print('Start data_streaker_offset for streaker %s, screen %s, beamline %s, dry_run %s' % (streaker, screen, beamline, dry_run))
    meta_dict_1 = get_meta_data()

    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    offset_pv = streaker+':CENTER'

    current_val = caget(offset_pv+'.RBV')

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
    print('End data_streaker_offset')
    return output

def move_pv(pv, value, timeout, tolerance):
    caput(pv, value)
    step_seconds = 0.1
    max_step = timeout // step_seconds
    for step in range(max_step):
        current_value = caget(pv+'.RBV')
        if abs(current_value - value) < tolerance:
            break
        else:
            time.sleep(step_seconds)
        if step % 10 == 0:
            caput(pv, value)
    else:
        raise ValueError('Pv %s should be %e, is: %e after %f seconds!' % (pv, value, current_value, timeout))


def bpm_data_streaker_offset(streaker, offset_range, screen, n_images, dry_run, beamline='Aramis'):
    print('Start bpm_data_streaker_offset for streaker %s, screen %s, beamline %s, dry_run %s' % (streaker, screen, beamline, dry_run))
    meta_dict_1 = get_meta_data()

    x_axis, y_axis = get_axis(screen)

    result_dict = {'x_axis_m': x_axis, 'y_axis_m': y_axis}

    offset_pv = streaker+':CENTER'

    # Start from closer edge of scan
    current_val = caget(offset_pv+'.RBV')
    if abs(current_val - offset_range[0]) > abs(current_val - offset_range[-1]):
        offset_range = offset_range[::-1]

    offset_range_mm = offset_range * 1e3

    result_dict['image'] = np.zeros([len(offset_range), n_images, len(y_axis), len(x_axis)], dtype=np.uint16)

    bpm_channels = config.beamline_bpm_pvs[beamline]

    for bpm_channel in bpm_channels:
        result_dict[bpm_channel] = np.zeros([len(offset_range), n_images])

    for n_offset, offset_mm in enumerate(offset_range_mm):
        if dry_run:
            print('I would move %s to %f' % (offset_pv, offset_mm))
            time.sleep(1)
        else:
            move_pv(offset_pv, offset_mm, 60, 1e-3)
        image_dict = get_images_and_bpm(screen, n_images, beamline, False, False, False, x_axis, y_axis, dry_run)['pyscan_result']
        for key in image_dict.keys():
            if key in result_dict:
                result_dict[key][n_offset] = image_dict[key]


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
    print('End bpm_data_streaker_offset')
    return output


def get_axis(screen):
    camera_client = CamClient()
    camera_stream_adress = camera_client.get_instance_stream(screen)
    host, port = get_host_port_from_stream_address(camera_stream_adress)

    with source(host=host, port=port, mode=SUB) as stream:
        data = stream.receive()
        print('Received image')

        x_axis = np.array(data.data.data['x_axis'].value)*1e-6
        y_axis = np.array(data.data.data['y_axis'].value)*1e-6
    return x_axis, y_axis


def get_images_and_bpm(screen, n_images, beamline='Aramis', axis=True, print_=True, include_meta_data=True, x_axis=None, y_axis=None, dry_run=False):

    if print_:
        print('Start get_images_and_bpm for screen %s, %i images, beamline %s' % (screen, n_images, beamline))

    if include_meta_data:
        meta_dict_1 = get_meta_data()
    else:
        meta_dict_1 = None

    result_dict = {}

    if axis:
        x_axis, y_axis = get_axis(screen)
        result_dict['x_axis_m'] = x_axis
        result_dict['y_axis_m'] = y_axis

    bpm_channels = config.beamline_bpm_pvs[beamline]
    image_pv = screen+':FPICTURE'
    channels = bpm_channels + [image_pv]

    images = np.zeros([n_images, len(y_axis), len(x_axis)], dtype=np.uint16)
    bpm_values = np.zeros([len(bpm_channels), n_images])
    pulse_ids = np.zeros(n_images)

    with source(channels=channels) as stream:
        for n_image in range(n_images):
            msg = stream.receive()
            pulse_ids[n_image] = msg.data.pulse_id
            for n_bpm, bpm_channel in enumerate(bpm_channels):
                try:
                    bpm_values[n_bpm, n_image] = msg.data.data[bpm_channel].value
                except KeyError:
                    print(msg.data.data.keys())
                    raise
            if not dry_run:
                images[n_image] = msg.data.data[image_pv].value

    result_dict['image'] = images
    result_dict['pulse_id'] = pulse_ids
    for n_bpm, bpm_channel in enumerate(bpm_channels):
        result_dict[bpm_channel] = bpm_values[n_bpm]

    # Configure bsread
    if include_meta_data:
        meta_dict_2 = get_meta_data()
    else:
        meta_dict_2 = None

    output_dict = {
            'pyscan_result': result_dict,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }

    if print_:
        print('End get_images')

    return output_dict



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
    meta_dict.update({x+':GAP': caget(x+':GAP.RBV') for x in all_streakers})
    meta_dict.update({x+':CENTER': caget(x+':CENTER.RBV') for x in all_streakers})

    k1l_dict = get_aramis_quad_strengths()
    meta_dict.update(k1l_dict)
    meta_dict['time'] = str(datetime.datetime.now())
    for gas_monitor_energy_pv in config.gas_monitor_pvs.values():
        meta_dict[gas_monitor_energy_pv] = caget(gas_monitor_energy_pv)
    return meta_dict

