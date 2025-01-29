# %%
# %% libraries
import pandas as pd
from numpy import abs
import numpy as np
import matplotlib.pyplot as plt

from scipy import fft

import datetime as dt
import os,sys

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_FILE_DIR)
sys.path.append(_SRC_DIR)

# local imports
import tsFB.data.prototyping_metrics as pm
import tsFB.utils.time_chunking as tc
import tsFB.build_filterbanks as fb
import tsFB.filterbank_analysis as fa

# %% [markdown]
# # Prepare test data

# %%
# %% Get data
year = '2006'
month = '01'
day = '01'

start_dt = dt.datetime.strptime(f'{year}-{month}-{day}','%Y-%m-%d')
end_dt = dt.datetime.strptime(f'{year}-{month}-{format(int(day)+1,'02')}','%Y-%m-%d')


test_cdf_file_path =_SRC_DIR+fa._OMNI_MAG_DATA_DIR+ year +'/omni_hro_1min_'+ year+month+'01_v01.cdf'
mag_df = fa.get_test_data(fname_full_path=test_cdf_file_path,
                           start_date=start_dt,
                           end_date=end_dt)
mag_df=mag_df['BX_GSE']

# %% Prepare FT of test data for Fourier applications
cadence = dt.timedelta(seconds=60)

# %% [markdown]
# # Detrending & Smoothing via Convolution in the Time Domain
# The Smoothing and Detrending code from "time_chunking.py" (not excutable here)
# 
# ```python
# mag_df=mag_df[cols]
# mag_df.sort_index(inplace=True)
# preprocessed_mag_df = mag_df.copy()
# 
# if detrend_window > timedelta(seconds=0):
#     LOG.debug('Detrending')
#     smoothed = preprocessed_mag_df.rolling(detrend_window,
#         center=True
#     ).mean()
#     # Subtract the detrend_window (e.g. 30 minutes or 1800s) to detrend
#     preprocessed_mag_df = preprocessed_mag_df - smoothed
# 
# if smooth_window > timedelta(seconds=0):
#     LOG.debug('Smoothing')
#     preprocessed_mag_df = preprocessed_mag_df.rolling(smooth_window,
#         center=True
#     ).mean()
# ```

# %%
SM_window = dt.timedelta(seconds=1100)
DT_window = dt.timedelta(seconds=2000)

# %% [markdown]
# # Theoretical Frequency Response
# 
# **(Formula from [Ch. 15 of *Digital Signal Processing Textbook*](https://www.dspguide.com/CH15.PDF)**)
# 
# Frequency response of an $M$ point moving average filter. The frequency, $f$, runs between $0$ and $0.5$. For $f = 0$, use $H[f] = 1$
# 
# $$H_M[f] = \frac{\sin(\pi f W_M)}{W_M\sin(\pi f)}$$

# %% [markdown]
# The following code is from the "build_filterbanks.py" `moving_avg_freq_response` function
# ```python
# def moving_avg_freq_response(f, window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
#     n = int(window.total_seconds()/cadence.total_seconds())
#     numerator = np.sin(np.pi*f*n)
#     denominator = n*np.sin(np.pi*f)

#     # deal with zero/zero
#     zidx = np.where(denominator==0.0)[0]
#     denominator[zidx] = numerator[zidx]= 1

#     return abs(numerator/denominator)```

# %% [markdown]
# ## Detrending in the frequency domain
# **JK's notation:** $\operatorname{FT}(x) = \widetilde{x}$
# 
# Since detrending in the time domain is `detrended_sig = signal - mvg_avg(window)`, the frequency response of detrending ($DT$) with detrending window, $W_d$, should be
# 
# \begin{align*}
# \widetilde{DT}[f] &= 1 - H_d[f] \\ 
# &= 1 - \frac{\sin(\pi f W_d)}{W_d \sin(\pi f)}
# \end{align*}

# %%
# Build theoretical frequency response 
data_len = mag_df.shape[0]  
freq_spectrum = np.linspace(0.0001,1/2,(data_len//2)+1)
DT_theory = fb.moving_avg_freq_response(f=freq_spectrum,
                                        window=DT_window,
                                        cadence=cadence)
DT_theory = 1 - DT_theory

# %% [markdown]
# ## Smoothing in the Frequency Domain
# Smoothing (SM), with smoothing window $W_s$, only uses the moving average frequency response formula directly. In other words,
# 
# \begin{align*}
# \widetilde{SM}[f] &= H_s[f] \\ 
# &= \frac{\sin(\pi f W_s)}{W_s \sin(\pi f)}
# \end{align*}
# 

# %%
SM_theory = fb.moving_avg_freq_response(f=freq_spectrum,
                                        window=SM_window,
                                        cadence=cadence)


# %%
FR_theory = DT_theory*SM_theory

# %%
mag_df.sort_index(inplace=True)
mag_df.interpolate(method='index', kind='linear',limit_direction='both',inplace=True)
df_index=pd.date_range(start=mag_df.index[0], end=mag_df.index[-1], freq=dt.timedelta(seconds=60))

sig_fft_df = fft.rfftn(mag_df,axes=0)

# %%
# Apply filter: theoretical frequency response
preprocessing = {'Detrended':DT_theory,'Smoothed':SM_theory,'Detrended + Smoothed':FR_theory}
filtered_y={}
for prepro in preprocessing.keys():
    Y = sig_fft_df.ravel()*preprocessing[prepro]
    filtered_y[prepro] = np.real(fft.irfft(Y))

# %%
fig,axes = plt.subplots(ncols=1,nrows=4,figsize=(12,8),sharex=True)

axes[0].plot(mag_df)
axes[0].set_title('Original signal')
axes[0].grid()
colors = ['tab:green','tab:orange','tab:purple']
for i,p in enumerate(filtered_y.keys()):
    axes[i+1].plot(mag_df.index[:-1],filtered_y[p],color=colors[i])
    axes[i+1].set_title(p)
    axes[i+1].grid()

plt.show()
# %%
# plot frequency response

fig, axes = plt.subplots(nrows=1,ncols=1)
axes.plot(freq_spectrum,SM_theory,linestyle='dotted',label='Smoothing')
axes.plot(freq_spectrum,DT_theory,linestyle='dotted',label='Detrending')
axes.plot(freq_spectrum,FR_theory,label='Detrend*Smooth')
axes.set_ylabel('Amplitude')
axes.set_xlabel("Frequency")
# axes.set_title("Smoothing & Detrending",fontsize=12)
axes.legend()


# %%
tri1 = fb.filterbank(data_len=mag_df.shape[0],
cadence=dt.timedelta(seconds=60))

# %%
for i,f in enumerate(SM_theory[:-1]):
            if f - SM_theory[i+1] <0:
                cnt_fr_idx = i
                break

# cnt_fr_idx = np.argmax(FR_theory)
uef = tri1.freq_spectrum['hertz'][cnt_fr_idx]

tri1.build_triangle_fb(filter_freq_range=(0,uef),
                       center_freq=[tri1.freq_spectrum['hertz'][np.argmax(FR_theory)]])

plt.plot(tri1.freq_spectrum['sample_rate_frac'],FR_theory,label='Detrend*Smooth')
plt.plot(tri1.freq_spectrum['sample_rate_frac'],tri1.fb_matrix.T,linestyle='dashdot')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
# %%
