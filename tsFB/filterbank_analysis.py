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

def get_test_data(fname_full_path=None,
                  fname = None,
                  instrument = 'omni',
                  start_date = dt.datetime(year=2019,month=5,day=15,hour=0),
                  end_date = dt.datetime(year=2019,month=5,day=16,hour=0),
                  rads_norm=True,
                  orbit_fname = None):
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
    if fname_full_path is None:
        if instrument == 'psp':
            data_dir = _PSP_MAG_DATA_DIR
        elif instrument=='wind' :
            data_dir = _WIND_MAG_DATA_DIR
        elif instrument == 'omni':
            data_dir = _OMNI_MAG_DATA_DIR

        assert fname is not None, "Need to provide value for fname or fname_full_path"
        # Generate the full path to the file
        fname_full_path = os.path.join(
            _SRC_DIR + data_dir,
            *fname.split('/') # this is required do to behavior of os.join
        )
        
    if instrument == 'psp':
        if orbit_fname is not None:
            orbit_fname = os.path.join(
            _SRC_DATA_DIR,
            orbit_fname)
            # orbit dataframe
            orbit = pd.read_csv(
                orbit_fname,
                sep=",",
                comment ="#",
                index_col='EPOCH_yyyy-mm-ddThh:mm:ss.sssZ',
                parse_dates=['EPOCH_yyyy-mm-ddThh:mm:ss.sssZ'],
            )
        mag_df = pm.read_PSP_dataset(
            fname=fname_full_path,
            orbit=orbit,
            rads_norm=rads_norm,
            exponents_list=_EXPONENTS_LIST
        )
    elif instrument == 'wind':
        mag_df = pm.read_WIND_dataset(
            fname=fname_full_path
        )
    elif instrument == 'omni':
        mag_df = pm.read_OMNI_dataset(
            fname=fname_full_path
        )
    mag_df.interpolate(inplace=True)
    mag_df = mag_df[start_date:end_date]


    return mag_df

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


def view_filter_decomposition(data,
                            fb_matrix,
                            fftfreq,
                            cadence = dt.timedelta(seconds=300),
                            figsize=(4,11),
                            gs_wspace = 0.2,
                            gs_hspace = 0.5,
                            xlim = None,
                            center_freq = None,
                            filterbank_plot_title='Filter bank',
                            orig_sig_plot_title='Original Signal',
                            plot_reconstruction=False,
                            plot_direct_residual = False,
                            plot_rel_residual=False,
                            abs_residual=True,
                            percent_rel_res = True,
                            res_eps = 0.01,
                            ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, nrows = fb_matrix.shape[0]+6,
                          figure = fig,
                          wspace=gs_wspace, hspace=gs_hspace)
    
    x = data.index
    y = data

    # Filterbank plot
    if xlim is None:
        xlim = (fftfreq[0],fftfreq[-1])
    ax = fig.add_subplot(gs[0:1])  
    ax.plot(fftfreq, fb_matrix.T)
    ax.grid(True)
    # ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim(xlim)
    ax.set_title(filterbank_plot_title)
    ax.set_xticks(center_freq)
    ax.tick_params(rotation=35,labelsize=8,axis='x')
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    
    # Filtered Signal Decomposition
    filtered_df = get_filtered_signals(data=data,
                                       fb_matrix=fb_matrix,
                                       fftfreq=fftfreq,
                                       cadence=cadence)
    gs2 = gridspec.GridSpecFromSubplotSpec(ncols=1,nrows=fb_matrix.shape[0]*2, subplot_spec=gs[6:-1],hspace=0)
    for i,bank in enumerate(filtered_df):

        ax0 = fig.add_subplot(gs2[2*i:2*i+2])    
        ax0.plot(bank)
        ax0.text(x=0.0,y=max(bank),s=f'center freq = {center_freq[i]:.2e}',
                 ha='left',va='top',
                #  fontweight='bold',
                fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        ax0.set_xticks([])
        ax0.set_yticks([])
        # ax0.set_ylim(min(filtered_df[i]),max(filtered_df[i])+(max(filtered_df[i])*0.5))

        if i==0:
            ax0.set_title('Signal decomposition',fontsize=15)
        
    # Original series
    gs1 = gridspec.GridSpecFromSubplotSpec(ncols = 1, nrows = 6, subplot_spec=gs[2:5],hspace=0)
    ax0 = fig.add_subplot(gs1[0:2])   
    ax0.plot(x, y,label='original')
    ax0.set_ylabel('(nT)')
    ax0.set_title(orig_sig_plot_title)
    ax0.tick_params(labelbottom=False)
    # ax0.set_xticks([])
    # ax0.set_yticks([])
    ax0.grid(True)

    # Reconstruction (on top of original)
    last_gs = 0
    if plot_reconstruction:
        ax0.plot(x,np.sum(filtered_df,axis=0),linestyle='dotted',alpha=0.9,label='filterbank reconstruction')
        ax0.legend(loc='upper right',bbox_to_anchor=(1.1, 1.2),fontsize=8)
    
    # Direct Residual
    if plot_direct_residual:
        res = get_reconstruction_residuals(filtered_df=filtered_df,
                                           real_signal=y,
                                           relative=False,
                                           percent=False,
                                           absolute=abs_residual)
        last_gs+=2
        ax1 = fig.add_subplot(gs1[last_gs:last_gs+2])
        ax1.plot(x,res)
        ax1.set_title('Direct Residual',y=1.0,pad=-14,
                    #   fontweight='bold',
                    fontsize=10,
                      bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        ax1.tick_params(labelbottom=False)
        ax1.set_ylabel('(nT)')
        ax1.set_ylim(min(res),max(res)+(max(res)*0.5))
        # ax1.get_xaxis().set_visible(False)
        # ax1.set_xticks([])
        ax1.grid(True)

    # Relative Residual
    if plot_rel_residual:
        rel_res = get_reconstruction_residuals(filtered_df=filtered_df,
                                           real_signal=y,
                                           relative=True,
                                           percent=percent_rel_res,
                                           absolute=abs_residual,
                                           epsilon=res_eps)
        last_gs+=2
        ax2 = fig.add_subplot(gs1[last_gs:last_gs+2])
        ax2.plot(x,rel_res)
        ax2.grid(True)
        ax2.set_ylim(min(rel_res),max(rel_res)+(max(rel_res)*0.5))
        ax2.tick_params(labelbottom=False)
        # ax2.set_xticks([])
        ax2.set_title('Relative Residual',y=1.0,pad=-14,
                    #   fontweight='bold',
                    fontsize=10,
                      bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        if percent_rel_res:
            ax2.set_ylabel('% error')
        ax2.set_xlabel('Time')
    plt.show()
    

if __name__ == '__main__':
    # args--------------------------------------------------
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
    # -------------------------------------------------------

    # Test data
    if args['input_file'] is None:
        year = str(args['start_year'])
        month = str(args['start_month'])
        test_cdf_file_path =_SRC_DIR+_OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'
    else:
        test_cdf_file_path = args['input_file']
    
    mag_df = get_test_data(fname_full_path=test_cdf_file_path,
                           start_date=args['start_date'],
                           end_date=args['stop_date'])
    # mag_df = mag_df-mag_df.mean()

    # Build Filterbank
    fltbnk = fb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
    fltbnk.build_triangle_fb(num_bands=4,
                        filter_freq_range=(0.0,0.001),
                        )
    fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
                         fftfreq=fltbnk.freq_spectrum['hertz'],
                         xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                         ylabel='Amplitude')
    fltbnk.add_DC_HF_filters()
    fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
                         fftfreq=fltbnk.freq_spectrum['hertz'],
                         xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                         ylabel='Amplitude')
    
    # Visualize application
    for col in mag_df.columns:
        view_filter_decomposition(data=mag_df[col],
                                     fb_matrix=fltbnk.fb_matrix,
                                     fftfreq=fltbnk.freq_spectrum['hertz'],
                                     cadence=dt.timedelta(minutes=1),
                                     xlim = (fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                                     center_freq = fltbnk.center_freq,
                                     orig_sig_plot_title=f'[{args["start_year"]}-{args['start_month']}-{args['start_day']}] Original series ({col})',
                                     plot_reconstruction=True,
                                     plot_direct_residual=True,
                                     plot_rel_residual=True,
                                     percent_rel_res=True,
                                     abs_residual=args['absolute_residual'],
                                     res_eps=args['residual_epsilon'])