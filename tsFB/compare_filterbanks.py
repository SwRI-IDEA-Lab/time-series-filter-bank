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
import tsFB.build_filterbanks as bfb
import tsFB.filterbank_analysis as fba

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

class compare_FB:
    def __init__(self,
                 data,
                 cadence:dt.timedelta,
                 windows:list):
        self.data = data
        self.data_len = len(data)
        self.cadence = cadence
        self.windows = windows

        # Moving Average Filterbank
        self.MA_fb = bfb.filterbank(data_len=self.data_len,
                                    cadence=self.cadence)
        self.MA_fb.build_DTSM_fb(windows=self.windows)
        self.MA_fb.add_mvgavg_DC_HF()
        
        # Mel (Triangle) Filterbank
        self.Tri_fb = bfb.filterbank(data_len=self.data_len,
                                     cadence=self.cadence)
        self.Tri_fb.build_triangle_fb((0.0,np.sort(self.MA_fb.center_freq)[-1]),
                                      center_freq=np.sort(self.MA_fb.center_freq[1:-1]))
        self.Tri_fb.add_DC_HF_filters()

        self.fb_analysis = None

        # other useful numbers
        self.n_filters = self.Tri_fb.fb_matrix.shape[0]
        self.freq_spec = self.Tri_fb.freq_spectrum
        self.center_freq = {}
        for unit in self.Tri_fb.freq_spectrum.keys():
            self.center_freq[unit] = self.Tri_fb.freq_spectrum[unit][self.Tri_fb.center_freq_idx]

    def build_comparative_analysis_tools(self,
                                        abs_residual=True,
                                        percent_rel_res=True,
                                        res_eps=0.01):

        # Analysis tools
        self.fb_analysis = {'Moving Average':{'fb_object':self.MA_fb,
                                              'fb_matrix':self.MA_fb.fb_matrix},
                            'Mel':{'fb_object':self.Tri_fb,
                                   'fb_matrix':self.Tri_fb.fb_matrix}}
        
        for i,fb_name in enumerate(self.fb_analysis.keys()):
            fltrbnk = self.fb_analysis[fb_name]
            # filtered signals
            filtered_df = fba.get_filtered_signals(data=self.data,
                                            fb_matrix=fltrbnk['fb_matrix'],
                                            fftfreq=self.freq_spec['hertz'],
                                            cadence=self.cadence)
            fltrbnk['filtered_sigs'] = filtered_df

            fltrbnk['reconstruction'] = np.sum(filtered_df,axis=0)
            # direct residuals
            res = fba.get_reconstruction_residuals(filtered_df=filtered_df,
                                            real_signal=self.data,
                                            relative=False,
                                            percent=False,
                                            absolute=abs_residual)
            fltrbnk['direct_residual'] = res
            
            # relative residuals
            rel_res = fba.get_reconstruction_residuals(filtered_df=filtered_df,
                                            real_signal=self.data,
                                            relative=True,
                                            percent=percent_rel_res,
                                            absolute=abs_residual,
                                            epsilon=res_eps)
            
            fltrbnk['relative_residual'] = rel_res
            
    def analyze_reconstruction(self,
                                abs_residual = True,
                                percent_rel_res =True,
                                res_eps=0.01,
                                figsize=(8,11),
                                orig_sig_plot_title='Original Signal',
                                plot_direct_residual=True,
                                plot_rel_residual=True):
        if self.fb_analysis is None:
            self.build_comparative_analysis_tools(abs_residual=abs_residual,
                                                  percent_rel_res=percent_rel_res,
                                                  res_eps=res_eps)
        x = self.data.index
        y = self.data

        # Original series
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(ncols = 1, nrows = 6,hspace=0)
        ax0 = fig.add_subplot(gs[0:2])   
        ax0.plot(x, y,color='black',label='original')
        ax0.set_ylabel('(nT)')
        ax0.set_title(orig_sig_plot_title)
        ax0.tick_params(labelbottom=False)
        ax0.grid(True)

        # Reconstruction (on top of original)
        last_gs = 0
        # Direct Residual
        if plot_direct_residual:
            last_gs+=2
            ax1 = fig.add_subplot(gs[last_gs:last_gs+2])
        if plot_rel_residual:
            last_gs+=2
            ax2 = fig.add_subplot(gs[last_gs:last_gs+2])

        l_colors = {'Moving Average': 'brown',
                    'Mel':'green'}

        for fb_name in self.fb_analysis.keys():
            fltrbank = self.fb_analysis[fb_name]
            ax0.plot(x,fltrbank['reconstruction'],linestyle='dotted',color=l_colors[fb_name],alpha=0.9,label=f'{fb_name} reconstruction')
            ax0.legend()#loc='upper right',bbox_to_anchor=(1.0, 1.2),fontsize=8)
            
            if plot_direct_residual:
                ax1.plot(x,fltrbank['direct_residual'],color=l_colors[fb_name],label=f'{fb_name}')
                ax1.set_title('Direct Residual',y=1.0,pad=-14,
                            #   fontweight='bold',
                            fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
                
                ax1.set_ylabel('(nT)')
                # ax1.set_ylim(min(dir_residuals[fb_name]),
                #                 max(dir_residuals[fb_name])+(max(dir_residuals[fb_name])*0.5))
                ax1.tick_params(labelbottom=False)
                ax1.grid(True)

            # Relative Residual
            if plot_rel_residual:
                ax2.plot(x,fltrbank['relative_residual'],color=l_colors[fb_name],label=f'{fb_name}')
                
                # ax2.set_ylim(min(rel_residuals[fb_name]),
                #                 max(rel_residuals[fb_name])+(max(rel_residuals[fb_name])*0.5))

                if percent_rel_res:
                    ax2.set_ylabel('% error')
                
                ax2.tick_params(labelbottom=False)
                ax2.set_title('Relative Residual',y=1.0,pad=-14,
                            #   fontweight='bold',
                            fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
                ax2.set_xlabel('Time')
                ax2.grid(True)
        plt.show()
        

    def compare_decomp(self,
                        freq_units:str=None,
                        figsize=(4,11),
                        gs_wspace = 0.1,
                        gs_hspace = 0.7,
                        orig_sig_plot_title='Original Signal',
                        abs_residual=True,
                        percent_rel_res = True,
                        res_eps = 0.01,
                        ):
        if freq_units is None:
            fftfreq = self.freq_spec['hertz']
        else:
            fftfreq = self.freq_spec[freq_units]

        if self.fb_analysis is None:
            self.build_comparative_analysis_tools(abs_residual=abs_residual,
                                                  percent_rel_res=percent_rel_res,
                                                  res_eps=res_eps)

        x = self.data.index
        y = self.data

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(ncols = 2, nrows = self.n_filters+5,
                            figure = fig,
                            wspace=gs_wspace, hspace=gs_hspace)

        # Original series
        ax0 = fig.add_subplot(gs[0:2,0:])   
        ax0.plot(x, y,color='black',label='original')
        ax0.set_ylabel('(nT)')
        ax0.set_title(orig_sig_plot_title)
        ax0.tick_params(labelbottom=False)
        ax0.grid(True)
        
        for i, fb_name in enumerate(self.fb_analysis.keys()):
            fltrbnk = self.fb_analysis[fb_name]
            fb_obj = self.fb_analysis[fb_name]['fb_object']
            # Filterbank plot
            xlim = (fb_obj.center_freq[0],fb_obj.center_freq[-1])
            ax = fig.add_subplot(gs[2:3,i])  
            ax.plot(fftfreq, fltrbnk['fb_matrix'].T)
            ax.grid(True)
            # ax.set_ylabel('Weight')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_xlim(xlim)
            ax.set_title(fb_name)
            ax.set_xticks(fb_obj.center_freq)
            ax.tick_params(rotation=35,labelsize=8,axis='x')
            ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
            if i==1:
                ax.tick_params(labelleft=False)
            
            # Plot Reconstruction
            ax1 = fig.add_subplot(gs[4:5,i])
            ax1.plot(x,fltrbnk['reconstruction'],color='darkblue')
            ax1.set_title(fb_name+' Signal Reconstruction')
            ax1.set_xticks([])
            ax1.grid()
            if i ==1:
                ax1.tick_params(labelleft=False)
        
            decomp_gs = gridspec.GridSpecFromSubplotSpec(ncols=1,nrows=self.n_filters*2, 
                                               subplot_spec=gs[5:,i],
                                               hspace=0)
            for j,bank in enumerate(fltrbnk['filtered_sigs']):
                ax2 = fig.add_subplot(decomp_gs[2*j:2*j+2])    
                ax2.plot(bank)
                if i ==1:
                    ax2.text(x=0,y=max(bank),s=f'center freq = {fb_obj.center_freq[j]:.2e}',
                            ha='right',va='top',
                            #  fontweight='bold',
                            fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
                    ax2.tick_params(labelleft=False)
                ax2.set_xticks([])
                ax2.set_yticks([])
                # ax1.set_ylim(min(filtered_df[j]),max(filtered_df[j])+(max(filtered_df[j])*0.5))

                if j==0:
                    ax2.set_title(fb_name+' Signal Decomposition',fontsize=12)

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
    
    mag_df = fba.get_test_data(fname_full_path=test_cdf_file_path,
                           start_date=args['start_date'],
                           end_date=args['stop_date'])
    # mag_df = mag_df-mag_df.mean()
    
    # Visualize application
    
    for col in mag_df.columns:
        comp_anly = compare_FB(data=mag_df[col],
                           cadence=dt.timedelta(seconds=60),
                           windows=[500,1000,2000,4000,8000])
        comp_anly.analyze_reconstruction(orig_sig_plot_title=f'[{args["start_year"]}-{args['start_month']}-{args['start_day']}] Original series ({col})',
                                         figsize=(8,5.5))
        comp_anly.compare_decomp(orig_sig_plot_title=f'[{args["start_year"]}-{args['start_month']}-{args['start_day']}] Original series ({col})',
                                     figsize=(8.5,11),
                                     percent_rel_res=True,
                                     abs_residual=args['absolute_residual'],
                                     res_eps=args['residual_epsilon'])