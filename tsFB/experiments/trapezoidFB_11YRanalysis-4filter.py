import cdflib

import argparse
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from numpy import abs, append, arange, insert, linspace, log10, round, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dill as pickle

from mpl_toolkits.mplot3d import Axes3D 

from scipy import fft
import random as rnd

from astropy.io import fits

import datetime as dt
import os,sys

_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# local imports
import tsFB.data.prototyping_metrics as pm
import tsFB.utils.time_chunking as tc
import tsFB.build_filterbanks as fb
import tsFB.data.helper_funcs as hf
import tsFB.filterbank_analysis as fa
import tsFB.visualization.visualization as fb_vis
import tsFB.utils.CR_dates as crdt

# Data paths
_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)

_EXPONENTS_LIST = [2.15, 1.05, 1.05]

# Debugger arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-input_file',
    default=None,
    help='direct path to file to use for test'
)
parser.add_argument(
    '-start_date',
    default=None,
    help='Start date for interval.'
    'If None, will use values from args `start_year`, `start_month`, and `start_day`'
)
parser.add_argument(
    '-stop_date',
    default=None,
    help='Stop date for interval. Defaults to 2018-12-31.'
)
parser.add_argument(
    '-start_year',
    default=None,
    help='Start year for interval.'
    'If None, value is randomized to value between 1994 and 2023.'
    'Defaults to None.'
)
parser.add_argument(
    '-start_month',
    default=None,
    help='Start month for interval.'
    'If None, value is randomized.'
    'Defaults to None.'
)
parser.add_argument(
    '-start_day',
    default=None,
    help='Start day for interval.'
    'If None, value is randomized to value between 1 and 28.'
    'Defaults to None.'
)
parser.add_argument(
    '-chunk_size',
    default=86400,
    help=(
        'Duration, in seconds, length of test data'
        'Defaults to 86400 seconds (1 day).'
    ),
    type=int
)
parser.add_argument(
    '-cadence',
    default=1,
    help=(
        'Final cadence of interpolated timeseries in seconds.'
        'Defaults to 1 second.'
    ),
    type=int
)
parser.add_argument(
    '-absolute_residual',
    help='Whether or not to use absolute value of residuals',
    default=True,
    action='store_true'
)
parser.add_argument(
    '-residual_epsilon',
    help='Epsilon in denominator of relative residual calculation (to minimize effect of dividing by near zero).',
    default=0.01,
    type=float
)

def list_of_strings(arg):
    return arg.split(',')

parser.add_argument('-cols', 
                    type=list_of_strings,
                    default=['B_mag','BX_GSE','BY_GSE','BZ_GSE'])

if __name__ == '__main__':
    # args==============================================================
    args = vars(parser.parse_args())
    if args['start_date'] is None:
        if args['start_year'] is None:
            args['start_year'] = rnd.randint(1981,2023)
        if args['start_month'] is None:
            args['start_month'] = format(rnd.randint(1,12),'02')
        if args['start_day'] is None:
            args['start_day'] = format(rnd.randint(1,28),'02')
        args['start_date'] = dt.datetime.strptime(
            f'{args['start_year']}-{args['start_month']}-{args['start_day']}',
            '%Y-%m-%d'
        )
    else:
        args['start_date'] = dt.datetime.strptime(
        args['start_date'],
        '%Y-%m-%d'
    )
        
    args['chunk_size'] = dt.timedelta(seconds=args['chunk_size'])

    if args['stop_date'] is None:
        args['stop_date'] = args['start_date'] + args['chunk_size']
    else:
        args['stop_date'] = dt.datetime.strptime(
            args['stop_date'],
            '%Y-%m-%d'
        )

    args['cadence'] = dt.timedelta(seconds=args['cadence'])
    

    # Test data=========================================================
    mag_df = fa.get_test_data(start_date=args['start_date'],
                               end_date=args['stop_date'],
                               cols=args['cols'])
    # mag_df = mag_df-mag_df.mean()

    # variables for 11years
    y11_freq = fb.time_window_to_npt_freq(dt.timedelta(days=365*11),
                                          data_cadence=dt.timedelta(minutes=1))

    # frequencies based on windows
    windows = [dt.timedelta(days=365*0.5),dt.timedelta(days=5),dt.timedelta(days=1),dt.timedelta(hours=18)]
    cntr_freq = [fb.time_window_to_npt_freq(w,data_cadence=dt.timedelta(minutes=1)) for w in windows]
    
    # variable for 1 day
    d1_freq = fb.time_window_to_npt_freq(dt.timedelta(hours=1.5),
                                         data_cadence=dt.timedelta(minutes=1))
    
    # Build Filterbank
    fltbnk = fb.filterbank(data_len=len(mag_df),
                           cadence=dt.timedelta(seconds=60))
    fltbnk.build_trapezoid_fb(filter_freq_range=None,
                             center_freq=None,
                             edge_freq=[y11_freq]+cntr_freq+[d1_freq],
                             freq_units='sample_rate_frac'
                             )
    fltbnk.add_DC_HF_filters()

    # synoptic map data
    cr_num = '2137'
    hdu_list = fits.open(f'/home/jkobayashi/gh_repos/time-series-filter-bank/data/FITS/IDSEAR_AIAsyn/aia193_synmap_cr{cr_num}.fits')
    image_data = hdu_list[0].data
    
    CR_dates = crdt.create_CR_date_dictionary('/home/jkobayashi/gh_repos/time-series-filter-bank/data/CR_Table.rdb.txt')
    cr_start,cr_end = crdt.get_start_end_dates(CR_dates=CR_dates,
                                     carr_rot_num=cr_num)

   
    scalar_params = ['F','flow_speed']
    mag_components = ['BX_GSE','BY_GSE','BZ_GSE']

    y_labs = {'scalar_params':['|B| (nT)','Flow Speed (km/s)'],
              'mag_components':['(nT)']}
    
    decomp_labels = ['Solar Cycle', 'Rotation','Transient','Denoise']

    fb_vis.filter_decomposition(data=mag_df[scalar_params],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(10,10),
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                #   sig_xlim=(cr_start,cr_end),
                                  y_labels= y_labs['scalar_params'],
                                  HR_ylim=[(-27,27),(-400,400)],
                                #   HRp1_ylim=True,
                                  center_freq = None,
                                  decomp_labels=decomp_labels,
                                  plot_reconstruction=False,
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False)
    
    fb_vis.filter_decomposition(data=mag_df[scalar_params],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  syn_map_data=image_data,
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(8.5,10),
                                  plot_filterbank=False,
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                  sig_xlim=(cr_start,cr_end),
                                  y_labels= y_labs['scalar_params'],
                                  HR_ylim=[(-8,8),(-90,90)],
                                #   HRp1_ylim=True,
                                  add_to_sig_title=f'(CR{cr_num})',
                                  center_freq = None,
                                  decomp_labels=decomp_labels,
                                  plot_reconstruction=False,
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False)
    
    
    # res1 = (dt.datetime(year=2008,month=12,day=16,hour=0),dt.datetime(year=2008,month=12,day=18,hour=12))

    # fb_vis.filter_decomposition(data=mag_df[scalar_params],
    #                               fb_matrix=fltbnk.fb_matrix,
    #                               fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
    #                               cadence=dt.timedelta(minutes=1),
    #                               figsize=(8.5,10),
    #                               plot_filterbank=False,
    #                             #   fb_xlim = (0,fltbnk.edge_freq[-1]),
    #                               sig_xlim= res1,
    #                               y_labels= y_labs['scalar_params'],
    #                               HR_ylim=[(-5,5),(-25,25)],
    #                               HRp1_ylim=True,
    #                               date_formatter="%m-%d %H:%M",
    #                               rotate_xticks=25,
    #                               center_freq = None,
    #                               plot_reconstruction=False,
    #                               fb_log_freq=True,
    #                               fb_plot_sci_not=False)
    
    
    # fb_vis.filter_decomposition(data=mag_df[mag_components],
    #                               fb_matrix=fltbnk.fb_matrix,
    #                               fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
    #                               cadence=dt.timedelta(minutes=1),
    #                               figsize=(8.5,10),
    #                               plot_filterbank=False,
    #                             #   fb_xlim = (0,fltbnk.edge_freq[-1]),
    #                               sig_xlim=res1,
    #                               y_labels= y_labs['mag_components'],
    #                               HR_ylim=-0.2,
    #                               date_formatter="%m-%d %H:%M",
    #                               rotate_xticks=15,
    #                               center_freq = None,
    #                               plot_reconstruction=False,
    #                               fb_log_freq=True,
    #                               fb_plot_sci_not=False)

    final_res = (dt.datetime(year=2013,month=6,day=1,hour=0),dt.datetime(year=2013,month=6,day=1,hour=12))
    fb_vis.filter_decomposition(data=mag_df[scalar_params],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(8.5,10),
                                  plot_filterbank=False,
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                  sig_xlim= final_res,
                                  ylim_eps=10,
                                  y_labels= y_labs['scalar_params'],
                                  HR_ylim=[(-10,10),(-150,150)],
                                  date_formatter="%m-%d %H:%M",
                                  rotate_xticks=25,
                                  center_freq = None,
                                  decomp_labels=decomp_labels,
                                  plot_reconstruction=False,
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False)

    fb_vis.filter_decomposition(data=mag_df[mag_components],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(8.5,10),
                                  plot_filterbank=False,
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                  sig_xlim=final_res,
                                  y_labels= y_labs['mag_components'],
                                  HR_ylim=-0.1,
                                  date_formatter="%m-%d %H:%M",
                                  rotate_xticks=15,
                                  center_freq = None,
                                  decomp_labels=decomp_labels,
                                  plot_reconstruction=False,
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False)
    
    fb_vis.filter_decomposition(data=mag_df[['BX_GSE','BY_GSE']],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(8.5,10),
                                  plot_filterbank=True,
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                  sig_xlim=final_res,
                                  y_labels=['BX (nT)','BY (nT)'],
                                  colors = ['turquoise','orange'],
                                  HR_ylim=[(-10,10),(-15,15)],
                                  HRp1_ylim=True,
                                  date_formatter="%m-%d %H:%M",
                                  rotate_xticks=10,
                                  center_freq = None,
                                  decomp_labels=decomp_labels,
                                  plot_reconstruction=False,
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False)