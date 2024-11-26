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

_FILE_DIR = os.path.dirname( os.path.abspath(__file__))
_MODEL_DIR = os.path.dirname(_FILE_DIR)
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# local imports
import tsFB.data.prototyping_metrics as pm
import tsFB.utils.time_chunking as tc

# Data paths
_PSP_MAG_DATA_DIR = '/sw-data/psp/mag_rtn/'
_WIND_MAG_DATA_DIR = '/sw-data/wind/mfi_h2/'
_OMNI_MAG_DATA_DIR = '/sw-data/nasaomnireader/'
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)

_EXPONENTS_LIST = [2.15, 1.05, 1.05]

# Debugger arguments
parser = argparse.ArgumentParser()


# %% Ch. 15 Formula
def moving_avg_freq_response(f,window=dt.timedelta(minutes=3000),cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)
    return abs(numerator/denominator)

def add_DC_HF_filters(fb_matrix,
                      DC = True,
                      HF = True):
    # DC
    if DC:
        DC_filter = 1-fb_matrix[0,:]
        minin = (DC_filter == np.min(DC_filter)).nonzero()[0][0]
        DC_filter[minin:] = 0
        fb_matrix = np.append(DC_filter[None,:], fb_matrix, axis=0)

    # HF
    if HF:
        HF_filter = 1-fb_matrix[-1,:]
        minin = (HF_filter == np.min(HF_filter)).nonzero()[0][0]
        HF_filter[0:minin] = 0
        fb_matrix = np.append(fb_matrix, HF_filter[None,:], axis=0)

    return fb_matrix

def visualize_filterbank(fb_matrix,
                         fftfreq,
                         xlim:tuple = None,
                         ylabel = 'Weight'):
    """Simple plot of filterbank"""
    fig,ax = plt.subplots(figsize=(8,3))
    ax.plot(fftfreq,fb_matrix.T)
    ax.grid(True)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_xlabel('Frequency  (Hz)')
    if xlim is None:
        xlim = (np.min(fftfreq),np.max(fftfreq))
    ax.set_xlim(xlim)
    
    plt.tight_layout()
    plt.show()

class filterbank:
    def __init__(self,
                 data_len:int,
                 cadence = dt.timedelta(seconds=60),
                 restore_from_file:str = None):
        self.data_len = data_len
        self.cadence = cadence
        self.freq_spectrum = np.linspace(0.0001,data_len/2,(data_len//2)+1)
        self.freq_smprt = np.linspace(0.0001,0.5,(data_len//2)+1)
        self.freq_hz_spec = self.freq_spectrum/(data_len*cadence.total_seconds())
        
        # placeholders
        self.fb_matrix = None
        self.edge_freq = None
        self.DC = False
        self.HF = False

        # if restore_from_file is not None:
        #     pkl = open(restore_from_file,'rb')
        #     fb_dict = pickle.load(pkl)
        #     self.fb_matrix = fb_dict['fb_matrix']
        #     self.fftfreq = fb_dict['fftfreq']
        #     self.edge_freq = fb_dict['edge_freq']
        #     self.DC = fb_dict['DC']
        #     self.HF = fb_dict['HF']

    def build_triangle_fb(self, 
                          filter_freq_range = (0,5),
                          num_bands = 2,
                          center_freq = None):
        """Creates filterbank matrix of triangle filters.

        Parameters
        ----------
        filter_freq_range : tuple
            (min_freq,max_freq)
            Minimum and maximum frequencies (in hz) that define the range in which the filters occupy.
            min_freq will be the first edge, and max_freq will be the last edge
            
        num_bands : int
            Number of filters in filter bank. 
            Used to evenly space out center frequencies across filter_freq_range.
            Not used if center_freq is not None
        center_freq : array
            Specified center frequencies of triangle filterbanks.
            If none or empty array, center_freq of the filterbank will be evenly spaced out using num_bands. 
        """
        freq_min, freq_max = filter_freq_range

        # if center frequencies not specified, centers are evenly spaced out given the 
        if center_freq is None or len(center_freq) == 0:
            delta_freq = abs(freq_max - freq_min) / (num_bands + 1.0)
            edge_freq = freq_min + delta_freq*arange(0, num_bands+2)
            center_freq = edge_freq[1:-1]
        else:
            if type(center_freq) == np.ndarray:
                center_freq = center_freq.tolist()
            edge_freq = [freq_min] + center_freq + [freq_max]
            num_bands = len(center_freq)
        
        lower_edges = edge_freq[:-2]
        upper_edges = edge_freq[2:]
        

        freqs = self.freq_hz_spec
        melmat = zeros((num_bands, len(freqs)))

        for iband, (center, lower, upper) in enumerate(zip(
                center_freq, lower_edges, upper_edges)):

            left_slope = (freqs >= lower)  == (freqs <= center)
            melmat[iband, left_slope] = (
                (freqs[left_slope] - lower) / (center - lower)
            )

            right_slope = (freqs >= center) == (freqs <= upper)
            melmat[iband, right_slope] = (
                (upper - freqs[right_slope]) / (upper - center)
            )
        self.fb_matrix = melmat 
        self.edge_freq = np.array(edge_freq)
        self.upper_edges = upper_edges
        self.center_freq = center_freq
        self.lower_edges = lower_edges

    def build_DTSM_fb(self,
                      windows = []):
        fb_matrix = zeros((len(windows)-1,len(self.freq_spectrum)))
        center_freq = []
        windows.sort(reverse=True)
        for i,w in enumerate(windows[:-1]):
            DT = 1 - moving_avg_freq_response(f=self.freq_smprt,
                                              window=dt.timedelta(seconds=w),
                                              cadence=self.cadence)
            SM = moving_avg_freq_response(f=self.freq_smprt,
                                          window=dt.timedelta(seconds=windows[i+1]),
                                          cadence=self.cadence)
            FR = SM*DT
            fb_matrix[i] = FR
            center_freq.append(self.freq_hz_spec[np.argmax(FR)])
        self.fb_matrix = fb_matrix
        self.center_freq = center_freq
        self.windows = windows
        

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        self.fb_matrix = add_DC_HF_filters(fb_matrix=self.fb_matrix,
                                           DC=DC,
                                           HF=HF)
        if DC:
            if self.center_freq[0] != self.edge_freq[0]:
                self.center_freq = np.insert(self.center_freq,0,self.edge_freq[0])
            if self.upper_edges[0] != self.edge_freq[1]:
                self.upper_edges = np.insert(self.upper_edges,0,self.edge_freq[1])
        if HF:
            if self.center_freq[-1] != self.edge_freq[-1]:
                self.center_freq = np.append(self.center_freq,self.edge_freq[-1])
            if self.lower_edges[-1] != self.edge_freq[-2]:
                self.lower_edges = np.append(self.lower_edges,self.edge_freq[-2])
        self.DC = DC
        self.HF = HF

    def add_mvgavg_DC_HF(self,
                         DC = True,
                         HF = True):
        if DC:
            SM = moving_avg_freq_response(f=self.freq_spectrum,
                                            window=dt.timedelta(seconds=max(self.windows)),
                                            cadence=self.cadence)
            self.fb_matrix = np.append(SM[None,:],self.fb_matrix,axis=0)
            self.DC = True
            cnt_fq = self.freq_hz_spec[np.argmax(SM)]
            if self.center_freq[0] != cnt_fq:
                self.center_freq = np.insert(self.center_freq,0,cnt_fq)
            # if self.center_freq[-1] != cnt_fq:
            #     self.center_freq = np.append(self.center_freq,cnt_fq)

        if HF:
            FR = moving_avg_freq_response(f=self.freq_spectrum,
                                            window=dt.timedelta(seconds=min(self.windows)),
                                            cadence=self.cadence)
            DT = 1 - FR
            self.fb_matrix = np.append(self.fb_matrix,DT[None,:],axis=0)
            self.HF = True
            for i,f in enumerate(FR[:-1]):
                if f - FR[i+1] <0:
                    cnt_fr_idx = i
                    break
            cnt_fq = self.freq_hz_spec[cnt_fr_idx]
            if self.center_freq[-1] != cnt_fq:
                self.center_freq = np.append(self.center_freq,cnt_fq)
            # if self.center_freq[0] != cnt_fq:
            #     self.center_freq = np.insert(self.center_freq,0,cnt_fq)


    def visualize_filterbank(self):
        """Show a plot of the built filterbank."""
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.freq_hz_spec,)
                            #  xlim=(self.edge_freq[0],self.edge_freq[-1]))

    # TODO: Update and fix filterbank saving with new updates (changed attributes, moving average FB, etc.)
    def save_filterbank(self):
        """Save the filterbank transformation matrix, fftfrequencies, and frequency endpoints 
        as a dictionary to a local pickle file"""
            
        filterbank_dictionary = {'fb_matrix': self.fb_matrix,
                                'fftfreq': self.fftfreq,
                                'edge_freq': self.edge_freq,
                                'center_freq': self.center_freq,
                                'lower_edges': self.lower_edges,
                                'upper_edges': self.upper_edges,
                                'DC': self.DC,
                                'HF': self.HF
                                }
            
       
        fb_prefix = f'fb'
        for edge in self.edge_freq:
            fb_prefix +=f'_{edge:.3e}'
        if self.DC:
            fb_prefix += '_DC'
        if self.HF:
            fb_prefix += '_HF'

        with open(_SRC_DATA_DIR + '/filterbanks/' + fb_prefix +'.pkl', 'wb') as f:
            pickle.dump(filterbank_dictionary,f)
