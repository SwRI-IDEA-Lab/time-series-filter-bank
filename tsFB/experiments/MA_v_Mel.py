# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft
from sklearn.metrics import r2_score

import argparse
import random as rnd
import datetime as dt
import os,sys

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# local imports
import tsFB.data.prototyping_metrics as pm
import tsFB.utils.time_chunking as tc
import tsFB.build_filterbanks as bfb
import tsFB.filterbank_analysis as fba
import tsFB.compare_filterbanks as cfb

# optional
import warnings
warnings.filterwarnings("ignore")

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
    default=60,
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

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
parser.add_argument(
    '-windows',
    help='List of running mean windows (in seconds).',
    default = [500,1000,2000,4000,8000],
    type=list_of_ints
)
parser.add_argument(
    '-freq_units',
    default=None,
    help='frequency units to use for x-axis of frequency domain plots'
)

parser.add_argument(
    '-res_eps',
    default=0.01,
    help='residual epsilon to minimize impacts of dividing by true signal values that are close to zero.'
)
def run_compare_exp(start_date=dt.datetime(year=2000,month=1,day=1,hour=0),
                    end_date=dt.datetime(year=2000,month=1,day=2,hour=0),
                    cadence = dt.timedelta(seconds=60),
                    windows=[500,1000,2000,4000,8000],
                    freq_units:str=None,
                    figsize=(4,11),
                    gs_wspace = 0.1,
                    gs_hspace = 0.7,
                    abs_residual=True,
                    percent_rel_res = True,
                    res_eps = 0.01
                    ):
    # Get data
    year_str = str(start_date.year)
    month_str = format(start_date.month,'02')
    day_str = format(start_date.day,'02')
    
    test_cdf_file_path =_SRC_DIR+fba._OMNI_MAG_DATA_DIR+ year_str +'/omni_hro_1min_'+ year_str+month_str+'01_v01.cdf'

    mag_df = fba.get_test_data(fname_full_path=test_cdf_file_path,
                            start_date=start_date,
                            end_date=end_date)
    
    
    for i,col in enumerate(mag_df.columns):
        comp_fb = cfb.compare_FB(data=mag_df[col],
                                cadence=cadence,
                                windows=windows)
        
        comp_fb.build_comparative_analysis_tools(abs_residual=True,
                                                percent_rel_res=True,
                                                res_eps=0.01)

        if i == 0:
            # plot all in single plot
            plt.figure(figsize=(10,5))
            for m_bank in comp_fb.MA_fb.fb_matrix:
                plt.plot(comp_fb.MA_fb.freq_spectrum['hertz'],m_bank,linewidth=2)
            for t_bank in comp_fb.Tri_fb.fb_matrix:
                plt.plot(comp_fb.Tri_fb.freq_spectrum['hertz'],t_bank,linestyle='dashdot')

            plt.xlim(0.0,comp_fb.Tri_fb.center_freq[-1])
            plt.xlabel('Frequency (Hz)')
            plt.grid()
            plt.title('Both filterbank types in single plot')
            plt.show()

            # Sum of filterbank amplitudes
            plt.plot(comp_fb.MA_fb.freq_spectrum['hertz'],np.sum(comp_fb.MA_fb.fb_matrix,axis=0),label=f'$\sum$ Moving Avg. filters ({comp_fb.MA_fb.fb_matrix.shape[0]} filters)')
            plt.plot(comp_fb.Tri_fb.freq_spectrum['hertz'],np.sum(comp_fb.Tri_fb.fb_matrix,axis=0),label='$\sum$ Mel filters')
            plt.xlabel('Frequency (hz)')
            plt.ylabel('Amplitude')
            plt.title('Sum of filter amplitudes across all frequencies')
            plt.legend()
            plt.show()


        # Signal Decomposition
        comp_fb.compare_decomp(freq_units=freq_units,
                            figsize=figsize,
                            gs_wspace=gs_wspace,
                            gs_hspace=gs_hspace,
                            orig_sig_plot_title=f'[{year_str}-{month_str}-{day_str}] Original Signal ({col})'
                            )

        # Signal reconstruction (residuals, etc.)
        comp_fb.analyze_reconstruction(abs_residual=abs_residual,
                                    percent_rel_res=percent_rel_res,
                                    res_eps=res_eps,
                                    orig_sig_plot_title=f'[{year_str}-{month_str}-{day_str}] Original Signal ({col})',
                                    plot_direct_residual=True,
                                    plot_rel_residual=True,
                                    figsize=(8,5.5))


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

    run_compare_exp(start_date=args['start_date'],
                    end_date=args['stop_date'],
                    cadence=args['cadence'],
                    windows=[500,1000,2000,4000,8000],
                    freq_units=args['freq_units'],
                    figsize=(8,11),
                    res_eps=args['res_eps']
                    )