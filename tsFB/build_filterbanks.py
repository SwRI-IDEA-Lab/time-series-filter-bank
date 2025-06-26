from numpy import abs, append, arange, insert, linspace, log10, round, zeros
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import dill as pickle

import datetime as dt
import os,sys

import argparse

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_MODEL_DIR)
sys.path.append(_SRC_DIR)

# Data paths
_SRC_DATA_DIR = os.path.join(_SRC_DIR,'data',)


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
                         ylabel = 'Weight',
                         freq_xlab_units=''):
    """Simple plot of filterbank"""
    fig,ax = plt.subplots(figsize=(8,3))
    ax.plot(fftfreq,fb_matrix.T)
    ax.grid(True)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_xlabel('Frequency'+freq_xlab_units)
    if xlim is None:
        xlim = (np.min(fftfreq),np.max(fftfreq))
    ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

def time_window_to_npt_freq(window:dt.timedelta,
                            data_cadence:dt.timedelta):
    n_pts = window.total_seconds()/data_cadence.total_seconds()
    return 1/n_pts

class filterbank:
    def __init__(self,
                 data_len:int,
                 cadence = dt.timedelta(seconds=60),
                 restore_from_file:str = None):
        self.data_len = data_len
        self.cadence = cadence
        # frequency spectrum (based on data length)----------------------------------------
        freq_sample_num = np.linspace(0.0,data_len//2,(data_len//2)+1).astype(np.int64)
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

        # placeholders---------------------------------------------------------------------
        self.fb_type = None
        self.fb_matrix = None
        self.edge_freq = None
        self.center_freq_idx = []
        # TODO: apply better SWE practice of using underscores for encapsulation of attributes
        self.DC = False
        self.HF = False

        # TODO: fix "restore_from_file"
        # if restore_from_file is not None:
        #     pkl = open(restore_from_file,'rb')
        #     fb_dict = pickle.load(pkl)
        #     self.fb_matrix = fb_dict['fb_matrix']
        #     self.fftfreq = fb_dict['fftfreq']
        #     self.edge_freq = fb_dict['edge_freq']
        #     self.DC = fb_dict['DC']
        #     self.HF = fb_dict['HF']

    def update_center_freq_idx(self,
                               freq_units = 'hertz'):
        spect = self.freq_spectrum[freq_units]
        current_lst = self.center_freq_idx
        # update_lst = np.where(np.isin(self.freq_spectrum[freq_units],self.center_freq))
        update_lst = []
        for k in self.center_freq:
            update_lst.append(np.argmin(np.abs(spect-k)))

        if not np.array_equiv(current_lst,update_lst):
            self.center_freq_idx = update_lst


    def build_triangle_fb(self, 
                          filter_freq_range = (0,5),
                          num_bands = 2,
                          center_freq = None,
                          freq_units='hertz'):
        """Creates filterbank matrix of triangle filters.
        
        Provide either `num_bands` or `center_freq` and ALWAYS provide `filter_freq_range`.
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
        self.fb_type = 'triangle'
        freq_min, freq_max = filter_freq_range # (trusting user to input correctly)

        # if center frequencies not specified, centers are evenly spaced out given the freq range
        if center_freq is None or len(center_freq) == 0:
            delta_freq = abs(freq_max - freq_min) / (num_bands + 1.0)
            edge_freq = freq_min + delta_freq*arange(0, num_bands+2)
            center_freq = edge_freq[1:-1]
        else:
            if type(center_freq) == np.ndarray:
                center_freq = center_freq.tolist()
            center_freq.sort()
            edge_freq = [freq_min] + center_freq + [freq_max]
            num_bands = len(center_freq)

        # Build triangle filters
        lower_edges = edge_freq[:-2]
        upper_edges = edge_freq[2:]

        freqs = self.freq_spectrum[freq_units]
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

        self.update_center_freq_idx(freq_units=freq_units)

    def build_trapezoid_fb(self,
                          filter_freq_range = (0.05,0.45),
                          center_freq = [(0.1,0.2),(0.3,0.4)], 
                          edge_freq = [0.05,0.1,0.2,0.3,0.4,0.45],
                          freq_units='sample_rate_frac'):
        """Creates filterbank matrix of trapezoidal filters.
        
        There are two ways to build the filter banks, either: 
            
            Provide both `filter_freq_range` (tuple) and `center_freq` (list of tuples), which indicates the range of frequencies
            the filters occupy and the list of tuples indicate the range for flat plateaus of each filter. 

            (This is a bit more deliberate approach, where the user knows which frequency ranges to use for each individual filter 
            and may specifically want to adjust the cut-offs of the DC & HF filters)
            
            OR
            
            (Enter `None` for both `filter_freq_range` and `center_freq` to utilize this method)

            Provide comprehensive `edge_freq` list, which includes all edge points of interest.
            If there are values entered in `filter_freq_range` and `center_freq` then anything passed in this `edge_freq` argument will be overwritten.

            (This is a more relaxed method to supply the entire list of frequencies of interest, without worrying about which are the trapezoid filters, etc.)
            
        ---------- 
        filter_freq_range : tuple
            (min_freq,max_freq)
            Minimum and maximum frequencies (in hz) that define the range in which the filters occupy.
            min_freq will be the first edge, and max_freq will be the last edge
        

        center_freq : list of tuples
            Specified center frequencies of triangle filterbanks.
            If none or empty array, center_freq of the filterbank will be evenly spaced out using num_bands. 

        edge_freq : list
            comprehensive list of all relevant edges at their associated frequency, including the starting and last edges
           
        """
        self.fb_type = 'trapezoid'

        # TODO: Add flexibility to just provide num_bands and filter_freq_range and automatically create evenly spaced filters (like in triangle filter bank function)
        if center_freq is None or len(center_freq) == 0:                # if center_freq is None, use provided edge_freq
            assert edge_freq is not None or len(edge_freq) == 0, "Either center_freq or edge_freq need to be provided, both cannot be None or empty."
            assert len(edge_freq)>=4 and len(edge_freq) % 2 ==0, "Even number of elements in edge_freq  is required (4 numbers minimum)"
            # TODO: (SWE good practice) add log message of using edge_freq list, and maybe even what the center frequencies are
            
            edge_freq.sort()

            freq_min = min(edge_freq)
            freq_max = max(edge_freq)
            filter_freq_range = (freq_min,freq_max)

            cl_edges = edge_freq[1:-1:2]
            cu_edges = edge_freq[2:-1:2]
            center_freq = []
            for i,l in enumerate(cl_edges):
                center_freq.append((l,cu_edges[i]))
            
        else: #(i.e. center frequencies are provided)
            # TODO: (SWE good practice) log message indicating list of center_freq is being used and that even if edge_freq is provided it will be overwritten
            freq_min, freq_max = filter_freq_range

            # TODO: (Current state: assuming perfect user that knows how to correctly provide list) Update to safeguard from improper provided center_freq list
            edge_freq = [freq_min]  # comprehensive list of edges
            cl_edges = []           # list of lower centers
            cu_edges = []           # list of upper centers
            for cntrs in center_freq:
                edge_freq.append(min(cntrs))
                cl_edges.append(min(cntrs))

                edge_freq.append(max(cntrs))
                cu_edges.append(max(cntrs))

            edge_freq.append(freq_max)

        num_bands = int((len(edge_freq)-2)/2)

        # Build trapezoidal filters
        lower_edges = edge_freq[:-2:2]
        upper_edges = edge_freq[3::2]

        freqs = self.freq_spectrum[freq_units]
        fltrmat = zeros((num_bands, len(freqs)))

        for iband, (lower, l_cntr, u_cntr, upper) in enumerate(zip(
                    lower_edges, cl_edges, cu_edges, upper_edges)):

            left_slope = (freqs >= lower)  == (freqs <= l_cntr)
            fltrmat[iband, left_slope] = (
                (freqs[left_slope] - lower) / (l_cntr - lower)
            )

            flat_slope = (freqs >= l_cntr) == (freqs <= u_cntr)
            fltrmat[iband, flat_slope] = 1 

            right_slope = (freqs >= u_cntr) == (freqs <= upper)
            fltrmat[iband, right_slope] = (
                (upper - freqs[right_slope]) / (upper - u_cntr)
            )
        self.fb_matrix = fltrmat 
        self.edge_freq = np.array(edge_freq)
        self.upper_edges = upper_edges
        self.cu_edges = cu_edges
        self.cl_edges = cl_edges
        self.center_freq = center_freq
        self.lower_edges = lower_edges

        # self.update_center_freq_idx(freq_units=freq_units)

    def build_DTSM_fb(self,
                      windows = []):
        self.fb_type = 'moving_average'
        fb_matrix = zeros((len(windows)-1,self.data_len//2+1))
        center_freq = []
        windows.sort(reverse=True)
        for i,w in enumerate(windows[:-1]):
            DT = 1 - moving_avg_freq_response(f=self.freq_spectrum['sample_rate_frac'],
                                              window=dt.timedelta(seconds=w),
                                              cadence=self.cadence)
            SM = moving_avg_freq_response(f=self.freq_spectrum['sample_rate_frac'],
                                          window=dt.timedelta(seconds=windows[i+1]),
                                          cadence=self.cadence)
            FR = SM*DT
            fb_matrix[i] = FR
            center_freq.append(self.freq_spectrum['hertz'][np.argmax(FR)])
        self.fb_matrix = fb_matrix
        self.center_freq = center_freq
        self.update_center_freq_idx()
        self.center_freq_idx = np.where(np.isin(self.freq_spectrum['hertz'],center_freq))
        self.windows = windows

    def add_DC_HF_filters(self,
                          DC = True,
                          HF = True):
        # DC
        if DC and not self.DC:
            # update fb_matrix
            DC_filter = 1-self.fb_matrix[0,:]
            minin = (DC_filter == np.min(DC_filter)).nonzero()[0][0]
            DC_filter[minin:] = 0
            self.fb_matrix = np.append(DC_filter[None,:], self.fb_matrix, axis=0)
            
            self.DC=True
            # update edge frequency lists
            if self.fb_type == 'triangle':
                if self.center_freq[0] != self.edge_freq[0]:
                    self.center_freq = np.insert(self.center_freq,0,self.edge_freq[0])
                if self.upper_edges[0] != self.edge_freq[1]:
                    self.upper_edges = np.insert(self.upper_edges,0,self.edge_freq[1])
        # HF
        if HF and not self.HF:
            # update fb_matrix
            HF_filter = 1-self.fb_matrix[-1,:]
            minin = (HF_filter == np.min(HF_filter)).nonzero()[0][0]
            HF_filter[0:minin] = 0
            self.fb_matrix = np.append(self.fb_matrix, HF_filter[None,:], axis=0)

            self.HF=True
            # update edge frequency lists
            if self.fb_type == 'triangle':
                if self.center_freq[-1] != self.edge_freq[-1]:
                    self.center_freq = np.append(self.center_freq,self.edge_freq[-1])
                if self.lower_edges[-1] != self.edge_freq[-2]:
                    self.lower_edges = np.append(self.lower_edges,self.edge_freq[-2])
        if self.fb_type == 'triangle':
            self.update_center_freq_idx()

    def add_mvgavg_DC_HF(self,
                         DC = True,
                         DC_flat_window = None,
                         DC_SM_window = None,
                         HF = True):
        #TODO: Finish trying to implement of flat top DC window (but should still work when DC does not include flattop portion)
        if DC and not self.DC:
            if DC_flat_window is not None:
                flat_n_w = DC_flat_window/self.cadence
                flat_f_width = 1/flat_n_w
                
            if DC_SM_window is None:
                DC_SM_window = max(self.windows)
            SM = moving_avg_freq_response(f=self.freq_spectrum['sample_rate_frac'],
                                            window=dt.timedelta(seconds=DC_SM_window),
                                            cadence=self.cadence)
            self.fb_matrix = np.append(SM[None,:],self.fb_matrix,axis=0)
            self.DC = True
            cnt_fq = self.freq_spectrum['hertz'][np.argmax(SM)]
            if self.center_freq[0] != cnt_fq:
                self.center_freq = np.insert(self.center_freq,0,cnt_fq)
            # if self.center_freq[-1] != cnt_fq:
            #     self.center_freq = np.append(self.center_freq,cnt_fq)

        if HF and not self.HF:
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
        self.update_center_freq_idx()

    def visualize_filterbank(self,
                             freq_units = 'hertz'):
        """Show a plot of the built filterbank."""
        visualize_filterbank(fb_matrix=self.fb_matrix,
                             fftfreq=self.freq_spectrum[freq_units],)
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

if __name__ == '__main__':
    import random as rnd
    import tsFB.filterbank_analysis as fa
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
    mag_df = fa.get_test_data(start_date=args['start_date'],
                               end_date=args['stop_date'],
                               cols=args['cols'])
    # mag_df = mag_df-mag_df.mean()

    # Build Filterbank
    fltbnk = filterbank(data_len=len(mag_df),
                           cadence=dt.timedelta(seconds=60))
    fltbnk.build_triangle_fb(filter_freq_range=(0.01,0.04),
                             center_freq=[0.02,0.03],
                             freq_units='sample_rate_frac'
                             )
    # fb.visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
    #                      fftfreq=fltbnk.freq_spectrum['hertz'],
    #                      xlim=(fltbnk.edge_freq[0],fltbnk.edge_freq[-1]),
    #                      ylabel='Amplitude')
    fltbnk.add_DC_HF_filters()
    visualize_filterbank(fb_matrix=fltbnk.fb_matrix,
                         fftfreq=fltbnk.freq_spectrum['sample_rate_frac'],
                         xlim=(fltbnk.edge_freq[0]-0.005,fltbnk.edge_freq[-1]+0.005),
                         ylabel='Amplitude',)
    
    import matplotlib.colors as mcolors
    for i,bank in enumerate(fltbnk.fb_matrix):
        fig,ax = plt.subplots(figsize=(8,3))
        ax.plot(fltbnk.freq_spectrum['sample_rate_frac'],bank,color = list(mcolors.TABLEAU_COLORS.keys())[i])
        ax.grid(True)
        ax.set_ylabel(ylabel='Amplitude')
        ax.set_xlabel('Frequency')
        # if xlim is None:
        #     xlim = (np.min(fftfreq),np.max(fftfreq))
        ax.set_xlim(0.0005,0.045)

        plt.tight_layout()
        plt.show()