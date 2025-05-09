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
import tsFB.visualization.visualization as fb_vis

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

def read_file(
        fname,
        mag_df=None,
        instrument='psp',
        rads_norm=True,
        orbit=None,
        cols=None
    ):
        """Read in the dataset and format it for input to SAX tree

        Parameters
        ----------
        fname : str
            The filename we wish to process
        instrument: str
            Solar wind instrument to analyze
        rads_norm : bool
            Boolean flag for indicating whether or not to perform radial normalization
        """
        # st_t = time.time()
        # self._current_file = fname
        if instrument == 'psp':
            data_dir = _PSP_MAG_DATA_DIR
        elif instrument=='wind' :
            data_dir = _WIND_MAG_DATA_DIR
        elif instrument == 'omni':
            data_dir = _OMNI_MAG_DATA_DIR

        # Generate the full path to the file
        fname_full_path = os.path.join(
            _SRC_DIR + data_dir,
            *fname.split('/') # this is required do to behavior of os.join
        )
        # LOG.debug(f'Extracting data from:\n {fname_full_path}')
        

        # TODO: Allow column selection flexibility to other instrument data reading functions (right now only OMNI is able to)
        if cols is None:
            if instrument == 'omni':
                cols = ['F','BX_GSE','BY_GSE','BZ_GSE']

        if instrument == 'psp':
            mag_df_new = pm.read_PSP_dataset(
                fname=fname_full_path,
                orbit=orbit,
                rads_norm=rads_norm,
                exponents_list=_EXPONENTS_LIST
            )
        elif instrument == 'wind':
            mag_df_new = pm.read_WIND_dataset(
                fname=fname_full_path
            )
        elif instrument == 'omni':
            mag_df_new = pm.read_OMNI_dataset(
                fname=fname_full_path,
                cols=cols
            )
        
        mag_df_new['filename'] = [fname]*mag_df_new.shape[0]
        
        if mag_df is not None:  
            # if self.mag_df is not empty, concat mag_df with existing self.mag_df
            mag_df_final = pd.concat([mag_df, mag_df_new])            
        else:        
            # otherwise, self.mag_df is not built yet and this is first self.mag_df
            mag_df_final = mag_df_new
        return mag_df_final

def get_test_data(instrument = 'omni',
                  start_date = dt.datetime(year=2019,month=5,day=15,hour=0),
                  end_date = dt.datetime(year=2019,month=5,day=16,hour=0),
                  rads_norm=True,
                  cols = ['B_mag','BX_GSE','BY_GSE','BZ_GSE']):
    """Retrieve a set of data to test and visualize filterbank application
    
    Parameters
    ----------
    fname_full_path : string
        complete file path to cdf file to extract data from
        A value for fname_full_path or fname (but not both) is required 
    fname : string
        part-way path to cdf file to extract data from, after 
        the selected "_DATA_DIR" that is selected by indicated instrument.
        A value for fname_full_path or fname (but not both) is required
    start_date: datetime, optional
        test data start time 
    end_date: datetime, optional
        test data end time
    rads_norm : bool, optional
        Boolean flag for controlling the normalization of the magnetic field 
        to account for the decay of the field strength with heliocentric distance
    orbit_fname : string, optional
        file path to psp orbit data
    """

    # Data catalog file name to access based on instrument
    if instrument == 'psp':
        catalog_fname = 'psp_master_catalog_2018_2021_rads_norm.csv' 
    elif instrument == 'wind':
        catalog_fname = 'wind_master_catalog_2006_2022.csv'
    elif instrument == 'omni':
        catalog_fname = 'data/B_FS_PD/omni_master_catalog_1994_2023.csv'

    catalog = pd.read_csv(
            _SRC_DIR+'/'+catalog_fname,
            index_col=0
        )
    if instrument == 'psp':
        fmt = '%Y%m%d%H'
    elif instrument == 'wind':
        fmt = '%Y%m%d'
    elif instrument == 'omni':
        fmt = '%Y%m%d'
    converter = lambda val: hf.fname_to_datetime(val, fmt=fmt)
    dates = catalog['fname'].apply(converter)
    catalog.index = pd.DatetimeIndex(dates, name='date')

    if start_date.day != 1:
        cat_start_dt = dt.datetime(year=start_date.year,month=start_date.month,day=1)
    else:
        cat_start_dt = start_date

    if end_date-start_date < dt.timedelta(weeks=4):
        cat_end_dt = start_date + dt.timedelta(weeks=5)
    else:
        cat_end_dt = end_date

    catalog_cut = catalog[cat_start_dt:cat_end_dt]
    flist = list(catalog_cut['fname'].values)
    
    mag_df=None
    for f in flist:
        mag_df = read_file(fname = f,
                            mag_df=mag_df,
                            rads_norm=rads_norm,
                            instrument=instrument,
                            cols=cols)
    
    mag_df.interpolate(inplace=True)
    if mag_df.iloc[0].isnull().any():
        mag_df.bfill(inplace=True)
    if mag_df.iloc[-1].isnull().any():
        mag_df.ffill(inplace=True)
    mag_df=mag_df[start_date:end_date]
    return mag_df[cols]

def get_filtered_signals(data,
                         fb_matrix,
                         fftfreq,
                         cadence):
    """
    
    Parameters
    ----------
    data : 1d array
        1d data array to pass through the filters
    
    fb_matrix : ndarray
        Matrix of filter bank filters

    fftfreq : ndarray
        Frequency spectrum

    cadence : dt.timedelta
        Cadence of data (inverse of sampling frequency)
    """
    filtered_df = np.zeros((fb_matrix.shape[0],data.shape[0]))

    # FFT
    sig_fft = fft.rfft(data) # the same as doing fft for each column

    for i,bank in enumerate(fb_matrix):
        filtered = sig_fft*bank
        f_sig = np.real(fft.irfft(filtered,data.shape[0]))
        filtered_df[i] = f_sig

    return filtered_df

def get_reconstruction_residuals(filtered_df,
                                 real_signal,
                                 epsilon=0.01,
                                 relative=True,
                                 absolute=True,
                                 percent=False
                                 ):
    reconstruction = np.sum(filtered_df,axis=0)
    residual = real_signal-reconstruction

    if absolute:
        residual=np.abs(residual)
        real_signal = np.abs(real_signal)

    if relative:
        if not absolute:
            epsilon=0
        rel_residual = residual/(real_signal+epsilon)
        if percent:
            return rel_residual*100
        return rel_residual
    
    return residual


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
    mag_df = get_test_data(start_date=args['start_date'],
                               end_date=args['stop_date'],
                               cols=args['cols'])
    # mag_df = mag_df-mag_df.mean()

    # variables for 11years
    y11_freq = fb.time_window_to_npt_freq(dt.timedelta(days=365*11),
                                          data_cadence=dt.timedelta(minutes=1))

    # frequencies based on windows
    windows = [dt.timedelta(days=365*0.5)]
    cntr_freq = [fb.time_window_to_npt_freq(w,data_cadence=dt.timedelta(minutes=1)) for w in windows]
    
    # variable for 1 day
    d1_freq = fb.time_window_to_npt_freq(dt.timedelta(days=1),
                                         data_cadence=dt.timedelta(minutes=1))

    # Build Filterbank
    fltbnk = fb.filterbank(data_len=len(mag_df),
                           cadence=dt.timedelta(seconds=60))
    fltbnk.build_triangle_fb(filter_freq_range=(y11_freq,d1_freq),
                             center_freq=cntr_freq,
                             freq_units='sample_rate_frac'
                             )
    # fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
    #                      fftfreq=fltbnk.freq_spectrum['hertz'],
    #                      xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
    #                      ylabel='Amplitude')
    fltbnk.add_DC_HF_filters()
    # fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
    #                      fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
    #                     #  xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
    #                      ylabel='Amplitude',)
    
    # Visualize application
    title_date_range = f'[{args["start_date"].year}-{format(args['start_date'].month,'02')}-{format(args['start_date'].day,'02')} to {args["stop_date"].year}-{format(args['stop_date'].month,'02')}-{format(args['stop_date'].day,'02')}]'
    for col in mag_df.columns:
        fb_vis.filter_decomposition(data=mag_df[col],
                                  fb_matrix=fltbnk.fb_matrix,
                                  fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                                  cadence=dt.timedelta(minutes=1),
                                  figsize=(11,8.5),
                                #   fb_xlim = (0,fltbnk.edge_freq[-1]),
                                  fb_log_freq=True,
                                  fb_plot_sci_not=False,
                                #   sig_xlim=(dt.datetime(year=2010,month=5,day=19),dt.datetime(year=2010,month=6,day=20)),
                                  center_freq = fltbnk.center_freq,
                                  orig_sig_plot_title=f'{title_date_range} Original series ({col})',
                                  plot_reconstruction=False)