from numpy import abs, append, arange, insert, linspace, log10, round, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dill as pickle

import datetime as dt
import os,sys

_MODEL_DIR = os.path.dirname( os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# Data paths
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)


def moving_avg_freq_response(f,
                             window=dt.timedelta(minutes=3000),
                             cadence=dt.timedelta(minutes=1)):
    n = int(window.total_seconds()/cadence.total_seconds())
    numerator = np.sin(np.pi*f*n)
    denominator = n*np.sin(np.pi*f)

    # deal with zero/zero
    zidx = np.where(denominator==0.0)[0]
    denominator[zidx] = numerator[zidx]= 1
    
    return abs(numerator/denominator)

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
        # frequency spectrum (based on data length)
        freq_sample_num = np.linspace(0.0,data_len/2,(data_len//2)+1)
        freq_sample_rate = freq_sample_num/data_len
        freq_natural = freq_sample_rate*2*np.pi
        if cadence is not None:
            freq_hz = freq_sample_num/(data_len*cadence.total_seconds())
        else:
            freq_hz = None

        self.freq_spectrum = {'sample_number':freq_sample_num,
                              'sample_rate_frac':freq_sample_rate,
                              'natural_frequency':freq_natural,
                              'hertz':freq_hz
                              }
        
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

        # if center frequencies not specified, centers are evenly spaced out given the freq range
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
        

        freqs = self.freq_spectrum['hertz']
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
        fb_matrix = zeros((len(windows)-1,self.data_len//2+1))
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
            center_freq.append(self.freq_spectrum['hertz'][np.argmax(FR)])
        self.fb_matrix = fb_matrix
        self.center_freq = center_freq
        self.windows = windows
        

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        # DC
        if DC:
            # update fb_matrix
            DC_filter = 1-self.fb_matrix[0,:]
            minin = (DC_filter == np.min(DC_filter)).nonzero()[0][0]
            DC_filter[minin:] = 0
            self.fb_matrix = np.append(DC_filter[None,:], self.fb_matrix, axis=0)

            # update edge frequency lists
            if self.center_freq[0] != self.edge_freq[0]:
                self.center_freq = np.insert(self.center_freq,0,self.edge_freq[0])
            if self.upper_edges[0] != self.edge_freq[1]:
                self.upper_edges = np.insert(self.upper_edges,0,self.edge_freq[1])
        # HF
        if HF:
            # update fb_matrix
            HF_filter = 1-self.fb_matrix[-1,:]
            minin = (HF_filter == np.min(HF_filter)).nonzero()[0][0]
            HF_filter[0:minin] = 0
            self.fb_matrix = np.append(self.fb_matrix, HF_filter[None,:], axis=0)

            # update edge frequency lists
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
            SM = moving_avg_freq_response(f=self.freq_spectrum['sample_rate_frac'],
                                            window=dt.timedelta(seconds=max(self.windows)),
                                            cadence=self.cadence)
            self.fb_matrix = np.append(SM[None,:],self.fb_matrix,axis=0)
            self.DC = True
            cnt_fq = self.freq_spectrum['hertz'][np.argmax(SM)]
            if self.center_freq[0] != cnt_fq:
                self.center_freq = np.insert(self.center_freq,0,cnt_fq)
            # if self.center_freq[-1] != cnt_fq:
            #     self.center_freq = np.append(self.center_freq,cnt_fq)

        if HF:
            FR = moving_avg_freq_response(f=self.freq_spectrum['sample_rate_frac'],
                                            window=dt.timedelta(seconds=min(self.windows)),
                                            cadence=self.cadence)
            DT = 1 - FR
            self.fb_matrix = np.append(self.fb_matrix,DT[None,:],axis=0)
            self.HF = True
            for i,f in enumerate(FR[:-1]):
                if f - FR[i+1] <0:
                    cnt_fr_idx = i
                    break
            cnt_fq = self.freq_spectrum['hertz'][cnt_fr_idx]
            if self.center_freq[-1] != cnt_fq:
                self.center_freq = np.append(self.center_freq,cnt_fq)
            # if self.center_freq[0] != cnt_fq:
            #     self.center_freq = np.insert(self.center_freq,0,cnt_fq)


    def visualize_filterbank(self):
        """Show a plot of the built filterbank."""
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.freq_spectrum['hertz'],)
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
