import numpy as np
import logging

import config

import pyscan
from cam_server import PipelineClient
from cam_server.utils import get_host_port_from_stream_address
from epics import caget; caget

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

def get_screen(screen, n_images, dry_run):
    time_interval = 1.
    print(time_interval, n_images)

    time_positioner = pyscan.TimePositioner(time_interval=time_interval, n_intervals=n_images)
    readables = [
            #'bs://gr_x_fit_standard_deviation',
            #'bs://gr_y_fit_standard_deviation',
            #'bs://gr_x_fit_mean',
            #'bs://gr_y_fit_mean',
            #'bs://gr_x_axis',
            #'bs://gr_y_axis',
            #'bs://gr_x_fit_gauss_function',
            #'bs://gr_y_fit_gauss_function',
            'bs://image',
            'bs://x_axis',
            'bs://y_axis',
            ]

    settings = pyscan.scan_settings(settling_time=0.01, measurement_interval=0.2, n_measurements=1)

    if dry_run:
        screen = 'simulation'

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

    output_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)
    return output_dict

def data_streaker_offset(streaker, offset_range, screen, n_images, dry_run):
    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    offset_pv = streaker+':CENTER'
    if dry_run:
        screen = 'simulation'
        writables = None
        positioner = pyscan.TimePositioner(time_interval=1, n_intervals=n_images)
    else:
        writables = [pyscan.epics_pv(pv_name=offset_pv, readback_pv_name=offset_pv+'.RBV', tolerance=0.05)]
        offset_range_mm = offset_range * 1e3
        positioner = pyscan.VectorPositioner(positions=(offset_range_mm).tolist())

    cam_instance_name = screen + '_sp1'

    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    logging.getLogger('mflow.mflow').setLevel(logging.ERROR)

    settings = pyscan.scan_settings(settling_time=0.5, measurement_interval=0.2, n_measurements=1)

    readables = [
            #'bs://gr_x_fit_standard_deviation',
            #'bs://gr_y_fit_standard_deviation',
            #'bs://gr_x_fit_mean',
            #'bs://gr_y_fit_mean',
            #'bs://gr_x_axis',
            #'bs://gr_y_axis',
            #'bs://gr_x_fit_gauss_function',
            #'bs://gr_y_fit_gauss_function',
            'bs://image',
            'bs://x_axis',
            'bs://y_axis',
            ]

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)
    output = [[x] for x in raw_output]

    result_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)
    all_streakers = config.all_streakers
    meta_dict = {}
    meta_dict.update({x+':GAP': caget(x+':GAP') for x in all_streakers})
    meta_dict.update({x+':CENTER': caget(x+':CENTER') for x in all_streakers})

    output = {
            'pyscan_result': result_dict,
            'streaker_offsets': offset_range,
            'screen': screen,
            'n_images': n_images,
            'dry_run': dry_run,
            'streaker': streaker,
            'meta_data': meta_dict,
            }

