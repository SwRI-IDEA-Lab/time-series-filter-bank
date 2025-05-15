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

    # frequencies based on windows
    windows3 = [dt.timedelta(days=365*11),
               dt.timedelta(days=365*0.5),dt.timedelta(days=5),
               dt.timedelta(days=1)]
    win3freq = [fb.time_window_to_npt_freq(w,data_cadence=dt.timedelta(minutes=1)) for w in windows3]
    

    windows4 = [dt.timedelta(days=365*11),
               dt.timedelta(days=365*0.5),dt.timedelta(days=5),
               dt.timedelta(days=1),dt.timedelta(hours=18),
               dt.timedelta(hours=1.5)]
    win4freq = [fb.time_window_to_npt_freq(w,data_cadence=dt.timedelta(minutes=1)) for w in windows4]
    
    # Build Filterbanks
    fltbnk3 = fb.filterbank(data_len=len(mag_df),
                           cadence=dt.timedelta(seconds=60))
    fltbnk3.build_trapezoid_fb(filter_freq_range=None,
                             center_freq=None,
                             edge_freq=win3freq,
                             freq_units='sample_rate_frac'
                             )
    fltbnk3.add_DC_HF_filters()

    fltbnk4 = fb.filterbank(data_len=len(mag_df),
                           cadence=dt.timedelta(seconds=60))
    fltbnk4.build_trapezoid_fb(filter_freq_range=None,
                             center_freq=None,
                             edge_freq=win4freq,
                             freq_units='sample_rate_frac'
                             )
    fltbnk4.add_DC_HF_filters()

   
    scalar_params = ['F','flow_speed']
    mag_components = ['BX_GSE','BY_GSE']

    y_labs = {'scalar_params':('|B| (nT)','Speed (km/s)'),
              'mag_components':('Bx (nT)','By (nT)')}
    
    deforest_res = (dt.datetime(year=2008,month=12,day=16,hour=0),dt.datetime(year=2008,month=12,day=18,hour=12))

    fb_vis.stack_subdecomp(data=mag_df[scalar_params+['proton_density','proton_density']+['T','T']+['BX_GSE','BY_GSE']+['BZ_GSE','BZ_GSE']],
                           main_filter=fltbnk3.fb_matrix[-1],
                           sub_filter=fltbnk4.fb_matrix[-2],
                           fftfreq=fltbnk3.freq_spectrum['sample_rate_frac'],
                           cadence=dt.timedelta(minutes=1),
                           figsize=(8.5,10),
                           sig_xlim=deforest_res,
                           main_alpha=0.3,
                           sub_alpha=0.9,
                           y_labels=[y_labs['scalar_params']]+[('Density (#/cm$^3$)','Density (#/cm$^3$)')]+[('Temperature (K)','Temperature (K)')]+[('Bx (nT)','By (nT)')]+[('Bz (nT)','Bz (nT)')],
                           colors=[('blue','red'),('purple','purple'),('green','green'),('teal','orange'),('orchid','orchid')],
                           ylims=[[(-5,5),(-25,25)],
                                  [(-15,15),(-15,15)],
                                  [(-25000,30000),(-25000,30000)],
                                  [(-3,3),(-5,5)],
                                  [(-5,6),(-5,6)]],
                           date_formatter="%m-%d %H:%M",
                           rotate_xticks=15)

    fb_vis.stack_subdecomp(data=mag_df[scalar_params+['proton_density','proton_density']+['T','T']+['BX_GSE','BZ_GSE']+['BY_GSE','BY_GSE']],
                           main_filter=fltbnk3.fb_matrix[-1],
                           sub_filter=fltbnk4.fb_matrix[-2],
                           fftfreq=fltbnk3.freq_spectrum['sample_rate_frac'],
                           cadence=dt.timedelta(minutes=1),
                           figsize=(8.5,10),
                           sig_xlim=deforest_res,
                           main_alpha=0.3,
                           sub_alpha=0.9,
                           y_labels=[y_labs['scalar_params']]+[('Density (#/cm$^3$)','Density (#/cm$^3$)')]+[('Temperature (K)','Temperature (K)')]+[('Bx (nT)','Bz (nT)')]+[('By (nT)','By (nT)')],
                           colors=[('blue','red'),('purple','purple'),('green','green'),('teal','orchid'),('orange','orange')],
                           ylims=[[(-5,5),(-25,25)],
                                  [(-15,15),(-15,15)],
                                  [(-25000,30000),(-25000,30000)],
                                  [(-3,3),(-5,5)],
                                  [(-5,6),(-5,6)]],
                           date_formatter="%m-%d %H:%M",
                           rotate_xticks=15)

    fb_vis.stack_subdecomp(data=mag_df[scalar_params+['proton_density','proton_density']+['T','T']+['BZ_GSE','BY_GSE']+['BX_GSE','BX_GSE']],
                           main_filter=fltbnk3.fb_matrix[-1],
                           sub_filter=fltbnk4.fb_matrix[-2],
                           fftfreq=fltbnk3.freq_spectrum['sample_rate_frac'],
                           cadence=dt.timedelta(minutes=1),
                           figsize=(8.5,10),
                           sig_xlim=deforest_res,
                           main_alpha=0.3,
                           sub_alpha=0.9,
                           y_labels=[y_labs['scalar_params']]+[('Density (#/cm$^3$)','Density (#/cm$^3$)')]+[('Temperature (K)','Temperature (K)')]+[('Bz (nT)','By (nT)')]+[('Bx (nT)','Bx (nT)')],
                           colors=[('blue','red'),('purple','purple'),('green','green'),('orchid','orange'),('teal','teal')],
                           ylims=[[(-5,5),(-25,25)],
                                  [(-15,15),(-15,15)],
                                  [(-25000,30000),(-25000,30000)],
                                  [(-5,6),(-5,6)],
                                  [(-3,3),(-3,3)]],
                           date_formatter="%m-%d %H:%M",
                           rotate_xticks=15)



    # Stacking in different order
    fb_vis.stack_subdecomp(data=mag_df[['flow_speed','T','F','proton_density','BY_GSE','BZ_GSE','BX_GSE','BX_GSE']],
                           main_filter=fltbnk3.fb_matrix[-1],
                           sub_filter=fltbnk4.fb_matrix[-2],
                           fftfreq=fltbnk3.freq_spectrum['sample_rate_frac'],
                           cadence=dt.timedelta(minutes=1),
                           figsize=(8.5,10),
                           sig_xlim=deforest_res,
                           main_alpha=0.3,
                           sub_alpha=0.9,
                           y_labels=[('Speed (km/s)','Temperature (K)'),
                                     ('|B| (nT)','Density (#/cm$^3$)'),
                                     ('By (nT)','Bz (nT)'),
                                     ('Bx (nT)','Bx (nT)')],
                           colors=[('red','green'),('blue','purple'),('orange','orchid'),('teal','teal')],
                           ylims=[[(-25,25),(-25000,30000)],
                                  [(-5,5),(-15,15)],
                                  [(-5,6),(-5,6)],
                                  [(-3,3),(-3,3)]],
                           date_formatter="%m-%d %H:%M",
                           rotate_xticks=15)
    
    fb_vis.stack_subdecomp(data=mag_df[['flow_speed','flow_speed','proton_density','proton_density','T','T','F','F','BY_GSE','BZ_GSE','BX_GSE','BX_GSE']],
                           main_filter=fltbnk3.fb_matrix[-1],
                           sub_filter=fltbnk4.fb_matrix[-2],
                           fftfreq=fltbnk3.freq_spectrum['sample_rate_frac'],
                           cadence=dt.timedelta(minutes=1),
                           figsize=(8.5,10),
                           sig_xlim=deforest_res,
                           main_alpha=0.3,
                           sub_alpha=0.9,
                           y_labels=[('Speed (km/s)','Speed (km/s)'),
                                     ('Density (cm$^3$)','Density (cm$^3$)'),
                                     ('Temp (K)','Temp (K)'),
                                     ('|B| (nT)','|B| (nT)'),
                                     ('By (nT)','Bz (nT)'),
                                     ('Bx (nT)','Bx (nT)')],
                           colors=[('red','red'),('purple','purple'),('green','green'),('blue','blue'),('orange','orchid'),('teal','teal')],
                           ylims=[[(-25,25),(-25,25)],
                                  [(-15,15),(-15,15)],
                                  [(-25000,30000),(-25000,30000)],
                                  [(-5,5),(-5,5)],
                                  [(-5,6),(-5,6)],
                                  [(-3,3),(-3,3)]],
                           date_formatter="%m-%d %H:%M",
                           rotate_xticks=10)