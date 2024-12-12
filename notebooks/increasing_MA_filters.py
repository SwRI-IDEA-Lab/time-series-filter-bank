# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft
from sklearn.metrics import r2_score

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

# optional
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# # Prepare test data

# %%
# %% Get data

year = '2019'
month = '06'
day = '01'
test_cdf_file_path =_MODEL_DIR+fba._OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'

mag_df = fba.get_test_data(fname_full_path=test_cdf_file_path,
                           start_date=dt.datetime(year=int(year),month=int(month),day=int(day),hour=0),
                           end_date=dt.datetime(year=int(year),month=int(month),day=int(day)+1,hour=0))
cols = 'BX_GSE'
mag_df=mag_df[[cols]]
mag_df
# mag_df = mag_df-mag_df.mean()  # Include this for fair comparison of DC and/or HF true or false (if both are true, original signal can be reconstructed regardless)

# %% Prepare FT of test data for Fourier applications

cadence = dt.timedelta(seconds=60)

mag_df.sort_index(inplace=True)
mag_df.interpolate(method='index', kind='linear',limit_direction='both',inplace=True)

real = np.array(mag_df).ravel()
# %% Excess power in relation to number of filters
winds = [500,1000,1500,2250]
lens1 = []
total_excess=[]
for i in range(25):
   
    DTSM = bfb.filterbank(data_len=len(mag_df),
                        cadence=dt.timedelta(seconds=60))
    # DTSM.build_DTSM_fb(windows=[500,1000,2000,5000,18000,21000,54000])
    DTSM.build_DTSM_fb(windows=winds)
    DTSM.add_mvgavg_DC_HF()

    lens1.append(DTSM.fb_matrix.shape[0])
    amp_sum = np.sum(abs(DTSM.fb_matrix),axis=0)
    total_excess.append(np.sum(amp_sum-1))

    plt.plot(DTSM.freq_spectrum['hertz'],amp_sum)
    plt.xlabel('Frequency (hz)')
    plt.title('Sum of filter amplitudes across all frequencies')
    plt.grid()

    winds.append(winds[-1]+(winds[-1]/2))
plt.show()
# %%
plt.figure()
plt.plot(lens1,total_excess)
plt.plot(lens1,total_excess,'o')
plt.title('Total Excess vs. Number of Filters')
plt.xlabel('Number of Filters')
plt.ylabel('Excess Amplitude (above 1)')
plt.grid()
plt.show()
# %%
winds = [500,1000,1500,2250]
lens2 = []
max_excess2=[]

for i in range(50):
   
    DTSM = bfb.filterbank(data_len=len(mag_df),
                        cadence=dt.timedelta(seconds=60))
    # DTSM.build_DTSM_fb(windows=[500,1000,2000,5000,18000,21000,54000])
    DTSM.build_DTSM_fb(windows=winds)
    DTSM.add_mvgavg_DC_HF()

    DTSM_filtered = fba.get_filtered_signals(data_df=mag_df,
                                                            fb_matrix=DTSM.fb_matrix,
                                                            fftfreq=DTSM.freq_spectrum['hertz'],
                                                            data_col=cols,
                                                            cadence=dt.timedelta(minutes=1),
                                                            )
    sum_DTSM_filtered = np.sum(DTSM_filtered,axis=0)

    lens2.append(DTSM.fb_matrix.shape[0])
    amp_sum = np.sum(abs(DTSM.fb_matrix),axis=0)
    max_excess2.append(np.sum(amp_sum-1))

    plt.plot(DTSM.freq_spectrum['hertz'],amp_sum)
    plt.xlabel('Frequency (hz)')
    plt.title('Sum of filter amplitudes across all frequencies')

    winds.append(winds[-1]*5)
plt.show()
# %%
plt.figure()
plt.plot(lens2,max_excess2)
plt.plot(lens2,max_excess2,'o',label='trial2')
plt.plot(lens1,total_excess,'o',label='trial1')
plt.title('Total Excess vs. Number of Filters')
plt.xlabel('Number of Filters')
plt.ylabel('Sum of Excess Amplitude (above 1)')
plt.legend()
plt.grid()
plt.show()
