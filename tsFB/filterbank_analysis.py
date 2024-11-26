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

def visualize_filterbank_application(data_df,
                                     fb_matrix,
                                     fftfreq,
                                     data_col = None,
                                     cadence = dt.timedelta(seconds=300),
                                     figsize=(24,8),
                                     wordsize_factor=2,
                                     gs_wspace = 0.2,
                                     gs_hspace = 0,
                                     xlim = None,
                                     center_freq = None,
                                     DC = False,
                                     HF = False,
                                     show_plot=True,
                                     save_results=False):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(ncols = 3, nrows = fb_matrix.shape[0]*2,
                          figure = fig,
                          wspace=gs_wspace, hspace=gs_hspace)
    
    if data_col is None:
        data_col = data_df.columns[-1]
    x = data_df.index
    y = data_df[data_col]
    all_filtered = np.zeros((fb_matrix.shape[0],data_df.shape[0]))

    data_span = x[-1]-x[0]

    
    for i in range(fb_matrix.shape[0]):
        # get filtered signal
        filtered_sig = tc.preprocess_fft_filter(mag_df=data_df,
                                                cols=data_df.columns,
                                                cadence=cadence,
                                                frequency_weights=fb_matrix[i,:],
                                                frequency_spectrum=fftfreq)
        
        filtered_sig = np.array(filtered_sig[data_col])
        
        all_filtered[i] = filtered_sig

        # wordsize calculation
        if HF and i == fb_matrix.shape[0]-1:
            word_size = int(0.9*len(x))
        else:
            if DC and i == 0:
                freq = (center_freq[1]-center_freq[0])/2
            else:
                freq = center_freq[i]
            word_size = int(wordsize_factor*data_span.total_seconds()*freq)

            if word_size > len(x):
                word_size = len(x)
        

        ax0 = fig.add_subplot(gs[2*i:2*i+2,1])    
        ax0.plot(x, filtered_sig,label=f'center_freq = {center_freq[i]:.2e}')
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.legend(loc='upper right',bbox_to_anchor=(1.4, 1))

        if i==0:
            ax0.set_title('Filter bank decomposition')
        
    if fb_matrix.shape[0]<5:
        os_gs = gs[3:4,0]
    else:
        os_gs = gs[4:6,0]


    ax0 = fig.add_subplot(os_gs)   
    ax0.plot(x, y)
    ax0.set_title('Original series')
    ax0.set_xticks([])
    ax0.set_yticks([])

    if xlim is None:
        xlim = (fftfreq[0],fftfreq[-1])
    ax = fig.add_subplot(gs[0:2,0])  
    ax.plot(fftfreq, fb_matrix.T)
    ax.grid(True)
    # ax.set_ylabel('Weight')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_xlim(xlim)
    ax.set_title('Mel filter bank')
    ax.set_xticks(center_freq)
    ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    if show_plot:
        plt.show()
    else:
        plt.close()

    if save_results:
        return all_filtered
    




if __name__ == '__main__':
    year = '2019'
    month = '05'
    test_cdf_file_path =_SRC_DIR+_OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

    mag_df = get_test_data(fname_full_path=test_cdf_file_path)
    mag_df = mag_df-mag_df.mean()
    # TODO: Set up json excutable way to test (using args, etc.)
    #=====================================
    # fltbnk = filterbank()
    # fltbnk.build_triangle_fb()
    # fltbnk.add_DC_HF_filters()
    # fltbnk.visualize_filterbank()
    #=====================================

    #=====================================
    fltbnk = fb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
    fltbnk.build_triangle_fb(num_bands=4,
                        filter_freq_range=(0.0,0.001),
                        )
    fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
                         fftfreq=fltbnk.freq_hz_spec,
                         xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                         ylabel='Amplitude')
    fltbnk.add_DC_HF_filters()
    # fltbnk.visualize_filterbank()
    fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
                         fftfreq=fltbnk.freq_hz_spec,
                         xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                         ylabel='Amplitude')
    #=====================================

    #=====================================
    # fltbnk = filterbank()
    # fltbnk.build_triangle_fb(num_bands=4,
    #                     sample_rate=1/60,
    #                     frequencies=[0.0,0.00025,0.00037,0.00065,0.000828,0.001],
    #                     num_fft_bands=int(1E6))
    # # fltbnk.add_DC_HF_filters()
    # fltbnk.visualize_filterbank()
    #=====================================

    #=====================================
    # fltbnk = filterbank(restore_from_file='/home/jkobayashi/gh_repos/idea-lab-sw-isax/data/filterbanks/fb_0.000e+00_1.250e-04_2.500e-04_3.750e-04_5.000e-04_6.250e-04_7.500e-04_8.750e-04_1.000e-03.pkl')
    # fltbnk.visualize_filterbank()
    #=====================================

    #=====================================
    # fltbnk = filterbank(data_len=len(mag_df),
    #                 cadence=dt.timedelta(seconds=60))
    # fltbnk.build_DTSM_fb(windows=[1000,3000,18000,108000])
    # fltbnk.visualize_filterbank()
    # fltbnk.add_mvgavg_DC_HF()
    # fltbnk.visualize_filterbank()
    # visualize_filterbank_application(data_df=mag_df,
    #                                  fb_matrix=fltbnk.fb_matrix,
    #                                  fftfreq=fltbnk.freq_hz_spec,
    #                                  data_col='BY_GSE',
    #                                  cadence=dt.timedelta(minutes=1),
    #                                  wordsize_factor = 3,
    #                                  xlim = (0,0.001),
    #                                  center_freq = fltbnk.center_freq,
    #                                  DC=fltbnk.DC,
    #                                  HF=fltbnk.HF)
    #=====================================

    visualize_filterbank_application(data_df=mag_df,
                                     fb_matrix=fltbnk.fb_matrix,
                                     fftfreq=fltbnk.freq_hz_spec,
                                     data_col='BY_GSE',
                                     cadence=dt.timedelta(minutes=1),
                                     wordsize_factor = 3,
                                     xlim = (fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
                                     center_freq = fltbnk.center_freq,
                                     DC=fltbnk.DC,
                                     HF=fltbnk.HF)