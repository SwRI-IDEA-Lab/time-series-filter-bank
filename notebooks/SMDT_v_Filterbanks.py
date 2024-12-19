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

# %% [markdown]
# # Summary of theoretical formulas
# 
# 
# ## General moving average filter
# **(Formula from [Ch. 15 of *Digital Signal Processing Textbook*](https://www.dspguide.com/CH15.PDF))**
# 
# Frequency response of an $M$ point moving average filter. 
# The frequency, $f$, runs between $0$ and $0.5$. For $f = 0$, use $H[f] = 1$
# 
# $$H[f] = \frac{\sin(\pi f M)}{M\sin(\pi f)}$$
# 
# ## Detrending ($DT$)
# Frequency response of detrending with window $W_d$,
# 
# \begin{align*}
# 
# \widetilde{DT}[f] &= 1 - H_d[f] \\
# 
# &= 1 - \frac{\sin (\pi f W_d)}{W_d \sin (\pi f)}
# 
# \end{align*}
# 
# ## Smoothing ($SM$)
# Frequency response of smoothing with window $W_s$,
# 
# \begin{align*}
# 
# \widetilde{SM}[f] &= H_s[f] \\
# 
# &= \frac{\sin (\pi f W_s)}{W_s \sin (\pi f)}
# 
# \end{align*}
# 
# ## *Both* detrending and smoothing 
# (just product of the above two)
# 
# \begin{align*}
# 
# \widetilde{DTSM}[f] &=  \widetilde{DT}[f] \cdot \widetilde{SM}[f] \\
# 
# &= (1-H_d[f]) \cdot (H_s[f]) \\
# 
# &= \left(1 - \frac{\sin (\pi f W_d)}{W_d \sin (\pi f)}\right) \cdot \left(\frac{\sin (\pi f W_s)}{W_s \sin (\pi f)}\right) 
# 
# \end{align*}

# %% [markdown]
# # Create Frequency Domain Filterbanks

# %% Moving average filterbanks
DTSM = bfb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
DTSM.build_DTSM_fb(windows=[500,1000,2000,4000,8000])

DTSM.add_mvgavg_DC_HF()
bfb.visualize_filterbank(fb_matrix=DTSM.fb_matrix,
                        fftfreq=DTSM.freq_spectrum['hertz'],
                        xlim=(0,DTSM.center_freq[-1]))

# %% triangle filterbanks
tri = bfb.filterbank(data_len=len(mag_df),
                    cadence=dt.timedelta(seconds=60))
tri.build_triangle_fb((0.0,np.sort(DTSM.center_freq)[-1]),
                      center_freq=np.sort(DTSM.center_freq[1:-1]))
tri.add_DC_HF_filters()
bfb.visualize_filterbank(fb_matrix=tri.fb_matrix,
                        fftfreq=tri.freq_spectrum['hertz'],
                        xlim=(0.0,tri.center_freq[-1]))

# %% plot all in single plot
plt.figure(figsize=(10,5))
for m_bank in DTSM.fb_matrix:
    plt.plot(DTSM.freq_spectrum['hertz'],m_bank,linewidth=2)
for t_bank in tri.fb_matrix:
    plt.plot(tri.freq_spectrum['hertz'],t_bank,linestyle='dashdot')

plt.xlim(0.0,tri.center_freq[-1])
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.title('Both filterbank types in single plot')
plt.show()

# %% Sum of filterbank amplitudes
plt.plot(DTSM.freq_spectrum['hertz'],np.sum(DTSM.fb_matrix,axis=0),label=f'$\sum$ Moving Avg. filters ({DTSM.fb_matrix.shape[0]} filters)')
plt.plot(tri.freq_spectrum['hertz'],np.sum(tri.fb_matrix,axis=0),label='$\sum$ Mel filters')
plt.xlabel('Frequency (hz)')
plt.ylabel('Amplitude')
plt.title('Sum of filter amplitudes across all frequencies')
plt.legend()
plt.show()


# %% [markdown]
# # Apply filterbanks to get a collection of deconstructed signals

# %%
DTSM_filtered = fba.get_filtered_signals(data=mag_df[cols],
                                         fb_matrix=DTSM.fb_matrix,
                                         fftfreq=DTSM.freq_spectrum['hertz'],
                                         cadence=dt.timedelta(minutes=1))
fba.view_filter_decomposition(data=mag_df[cols],
                            fb_matrix=DTSM.fb_matrix,
                            fftfreq=DTSM.freq_spectrum['hertz'],
                            cadence=dt.timedelta(minutes=1),
                            xlim = (0,DTSM.center_freq[-1]),
                            center_freq = DTSM.center_freq,
                            filterbank_plot_title='Moving Average Filter Bank',
                            orig_sig_plot_title=f'[{year}-{month}-{day}] Original Signal ({cols})',
                            plot_reconstruction=True,
                            plot_direct_residual=True,
                            plot_rel_residual=True,
                            percent_rel_res=True,
                            abs_residual=True,
                            res_eps=0.05)

# %%
tri_filtered = fba.get_filtered_signals(data=mag_df[cols],
                                        fb_matrix=tri.fb_matrix,
                                        fftfreq=tri.freq_spectrum['hertz'],
                                        cadence=dt.timedelta(minutes=1))
fba.view_filter_decomposition(data=mag_df[cols],
                            fb_matrix=tri.fb_matrix,
                            fftfreq=tri.freq_spectrum['hertz'],
                            cadence=dt.timedelta(minutes=1),
                            xlim = (0,tri.center_freq[-1]),
                            center_freq = tri.center_freq,
                            filterbank_plot_title='Mel Filter bank',
                            orig_sig_plot_title=f'[{year}-{month}-{day}] Original Signal ({cols})',
                            plot_reconstruction=True,
                            plot_direct_residual=True,
                            plot_rel_residual=True,
                            percent_rel_res=True,
                            abs_residual=True,
                            res_eps=0.05)

# %% [markdown]
# ## "bank" of filtered signals from applying smoothing & detrending in time domain
# Manually create collection of filtered signals from doing convolution in time domain with the same windows.
# %% "filterbank" of Smoothing & detrending via convolution
convolution_filtered = np.zeros(DTSM_filtered.shape)
DTSM.windows.sort(reverse=True)
for i,w in enumerate(DTSM.windows[:-1]):
    filtered = tc.preprocess_smooth_detrend(mag_df=mag_df,
                                            cols=cols,
                                            detrend_window=dt.timedelta(seconds=w),
                                            smooth_window=dt.timedelta(seconds=DTSM.windows[i+1]))
    convolution_filtered[i+1] = np.array(filtered).ravel()
# DC
if DTSM.DC:
    DC_filtered = tc.preprocess_smooth_detrend(mag_df=mag_df,
                                            cols=cols,
                                            detrend_window=dt.timedelta(seconds=0),
                                            smooth_window=dt.timedelta(seconds=max(DTSM.windows)))
    convolution_filtered[0] = np.array(DC_filtered).ravel()
# HF
if DTSM.HF:
    HF_filtered = tc.preprocess_smooth_detrend(mag_df=mag_df,
                                            cols=cols,
                                            detrend_window=dt.timedelta(seconds=min(DTSM.windows)),
                                            smooth_window=dt.timedelta(seconds=0))
    convolution_filtered[-1] = np.array(HF_filtered).ravel()


# %% [markdown]
# # Compare reconstructed signals (by summing all filtered signals)
# Sum up all of the filtered signals that came out of each filterbank, and compare how they did

# %% save sum of signals
sum_DTSM_filtered = np.sum(DTSM_filtered,axis=0)
sum_tri_filtered = np.sum(tri_filtered,axis=0)
sum_conv_filtered = np.sum(convolution_filtered,axis=0)

# %% [markdown]
# ## Compare signals directly

# %% calculate r-squared scores
# TODO: these scores may not be that useful, so can probably get rid of them
real = np.array(mag_df).ravel()
DTSM_r2 = r2_score(real,sum_DTSM_filtered)
tri_r2 = r2_score(real,sum_tri_filtered)
conv_r2 = r2_score(real,sum_conv_filtered)

# %% Plot Original vs. Convolution vs. Triangles
plt.figure(figsize=(10,5))
plt.plot(mag_df.index,real,color='black',label='original data')
# plt.plot(mag_df.index,sum_conv_filtered,linestyle='dashed',label=f'Convolution reconstruction ($R^2$: {conv_r2:.2f})')
plt.plot(mag_df.index,sum_DTSM_filtered,color='tab:orange',alpha=0.7,label=f'DTSM reconstruction ($R^2$:{DTSM_r2:.2e})')
plt.plot(mag_df.index,sum_tri_filtered,color='tab:green',linestyle='dotted',alpha=0.5,label=f'triangle reconstruction ($R^2$: {tri_r2:.2f})')
plt.title('Compare reconstructed signals directly with original')
plt.xlabel('Date & Time (MM-DD-HH)')
plt.ylabel('Magnetic field (nT)')
plt.legend()
plt.show()

# %% Plot Original vs. Convolution vs. Triangles
# sums = {'DTSM':sum_DTSM_filtered,
#         'Convolution': sum_conv_filtered,
#         'Triangles':sum_tri_filtered}
# for i,selection in enumerate(['DTSM','Convolution','Triangles']):
#     plt.figure(figsize=(10,5))
#     plt.plot(mag_df.index,real,label='original data')
#     plt.plot(mag_df.index,sums[selection],linestyle='dashed',label=f'$\sum$ {selection}')
#     plt.title('Compare reconstructed signals directly with original')
#     plt.xlabel('Date & Time (MM-DD-HH)')
#     plt.ylabel('Magnetic field (nT)')
#     plt.legend()
#     plt.show()


# %% residuals
DTSM_residual = fba.get_reconstruction_residuals(filtered_df=DTSM_filtered,
                                                 real_signal=real,
                                                 relative=False,
                                                 absolute=False)
tri_residual = fba.get_reconstruction_residuals(filtered_df=tri_filtered,
                                                real_signal=real,
                                                relative=False,
                                                absolute=False)
conv_residual = fba.get_reconstruction_residuals(filtered_df=convolution_filtered,
                                                 real_signal=real,
                                                 relative=False,
                                                 absolute=False)

residuals = {'DTSM':DTSM_residual,
            'Triangles':tri_residual,
            'Convolution':conv_residual}

# %% [markdown]
# ## Residuals of reconstructed signals compared with original
# %% Plot residuals: all in single plot
plt.figure(figsize=(10,5))
for i,selection in enumerate(['DTSM','Convolution','Triangles']):
    plt.plot(mag_df.index,residuals[selection],
             label=f'{selection} (total: {sum(abs(residuals[selection])):.2e}; Average: {abs(residuals[selection].mean()):.2e})')
    plt.legend()
plt.xlabel('Index')
plt.ylabel('Residual (nT)')
plt.title('Direct residual of reconstructed signals compared with original data signal')
plt.grid()
plt.show()


# %% relative residuals
DTSM_rel_residual = fba.get_reconstruction_residuals(filtered_df=DTSM_filtered,
                                                     real_signal=real,
                                                     relative=True,
                                                     percent=True,
                                                     absolute=False)
tri_rel_residual = fba.get_reconstruction_residuals(filtered_df=tri_filtered,
                                                    real_signal=real,
                                                    relative=True,
                                                    percent=True,
                                                    absolute=False)
conv_rel_residual = fba.get_reconstruction_residuals(filtered_df=convolution_filtered,
                                                     real_signal=real,
                                                     relative=True,
                                                     percent=True,
                                                     absolute=False)

rel_residuals = {'DTSM':DTSM_rel_residual,
            'Triangles':tri_rel_residual,
            'Convolution':conv_rel_residual}

# %% Plot relative residuals
plt.figure(figsize=(10,5))
for i,selection in enumerate(['DTSM','Convolution','Triangles']):
    plt.plot(mag_df.index,rel_residuals[selection],
             label=f'{selection}')
    plt.legend()
plt.xlabel('Index')
plt.ylabel('Relative Residual (%)')
plt.title('Relative Residual of reconstructed signals compared with original data signal')
plt.grid()
plt.show()
