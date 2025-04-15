import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from astropy.io import fits

import pandas as pd

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
import tsFB.data.helper_funcs as hf
import tsFB.filterbank_analysis as fa

def filter_decomposition(data,
                            fb_matrix,
                            fftfreq,
                            cadence = dt.timedelta(seconds=300),
                            figsize=(4,11),
                            gs_wspace = 0.2,
                            gs_hspace = 0.0,
                            fb_xlim = None,
                            sig_xlim = None,
                            center_freq = None,
                            filterbank_plot_title='Filter bank',
                            fb_freq_units = '',
                            fb_log_freq = False,
                            fb_plot_sci_not = True,
                            orig_sig_plot_title='Original Signal',
                            plot_reconstruction=False,
                            ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    x = data.index
    y = data

    gs_recon = 3 if plot_reconstruction else 0
    # initialize gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = 3+3+gs_recon+fb_matrix.shape[0]*2,
                           figure = fig,
                           wspace=gs_wspace, 
                           hspace=gs_hspace)
    
    # Original series
    ax0 = fig.add_subplot(gs[0:2])   
    ax0.plot(data,color='black',label='original')
    ax0.set_ylabel('(nT)')
    ax0.set_title(f'Original Series [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}]')
    ax0.grid(True)

    # Filterbank plot
    ax1 = fig.add_subplot(gs[3:4])  
    if fb_xlim is not None:
        ax1.set_xlim(fb_xlim)
    ax1.plot(fftfreq, fb_matrix.T)
    ax1.grid(True)
    if fb_log_freq:
        ax1.set_xscale('log')
        fb_freq_units += ' [log scaled]'
    ax1.set_xlabel('Frequency'+fb_freq_units)
    
    ax1.set_title(filterbank_plot_title)
    # ax1.set_xticks(center_freq)
    ax1.tick_params(rotation=35,labelsize=8,axis='x')
    if fb_plot_sci_not:
        ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    

    # Reconstruction (on top of original)
    if plot_reconstruction:
        ax2 = fig.add_subplot(gs[6:8],sharex=ax0,sharey=ax0)
        ax2.plot(x,np.sum(filtered_df,axis=0),linestyle='dotted',alpha=0.9,label='filterbank reconstruction')
        ax2.legend(loc='upper right',bbox_to_anchor=(1.1, 1.2),fontsize=8)
        last_gs = 9
    else:
        last_gs = 6
    
    # Filtered Signal Decomposition
    filtered_df = fa.get_filtered_signals(data=data,
                                       fb_matrix=fb_matrix,
                                       fftfreq=fftfreq,
                                       cadence=cadence)
    for i,bank in enumerate(filtered_df):
        ax3 = fig.add_subplot(gs[last_gs+2*i:last_gs+2*i+2],sharex=ax0)    
        ax3.plot(x,bank)
        if center_freq is not None:
            ax3.text(x=min(x),y=max(bank),s=f'center freq = {center_freq[i]:.2e}',
                 ha='left',va='top',
                 fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        if i != filtered_df.shape[0]-1:
            ax3.tick_params(labelbottom=False)
        ax3.grid(True)
        if i==0:
            ax3.set_title('Signal decomposition',fontsize=15)
    if sig_xlim is not None:
        ax3.set_xlim(sig_xlim)
    plt.show()
    
def filter_decomposition_with_synoptic(data,
                                       fb_matrix,
                                       fftfreq,
                                       syn_map_data,
                                       cadence = dt.timedelta(seconds=300),
                                       figsize=(4,11),
                                       gs_wspace = 0.2,
                                       gs_hspace = 0.0,
                                       fb_xlim = None,
                                       sig_xlim = None,
                                       center_freq = None,
                                       filterbank_plot_title='Filter bank',
                                       fb_freq_units = '',
                                       fb_log_freq = False,
                                       fb_plot_sci_not = True,
                                    #    orig_sig_plot_title='Original Signal',
                                       plot_reconstruction=False,
                                       syn_map_cmap = 'gray'
                                       ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    x = data.index
    y = data

    orig_sig_plot_title = f'Original Series [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}]'
    if sig_xlim is not None:
        orig_sig_plot_title = f'Original Series [{sig_xlim[0].strftime('%Y-%m-%d')} to {sig_xlim[-1].strftime('%Y-%m-%d')}]'

    gs_recon = 3 if plot_reconstruction else 0
    # initialize gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = 3+3+gs_recon+fb_matrix.shape[0]*2+5, # original series + filterbank plot + reconstruction plot (if applicable) + decomposition
                           figure = fig,
                           wspace=gs_wspace, 
                           hspace=gs_hspace)
    
    # Original series
    ax0 = fig.add_subplot(gs[0:2])   
    ax0.plot(data,color='black',label='original')
    ax0.set_ylabel('(nT)')
    ax0.set_title(orig_sig_plot_title)
    ax0.grid(True)

    # Filterbank plot
    if fb_xlim is not None:
        ax1.set_xlim(fb_xlim)
    ax1 = fig.add_subplot(gs[3:4])  
    ax1.plot(fftfreq, fb_matrix.T)
    ax1.grid(True)
    if fb_log_freq:
        ax1.set_xscale('log')
        fb_freq_units += ' [log scaled]'
    ax1.set_xlabel('Frequency'+fb_freq_units)
    
    ax1.set_title(filterbank_plot_title)
    # ax1.set_xticks(center_freq)
    if fb_plot_sci_not:
        ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        ax1.tick_params(rotation=35,labelsize=8,axis='x')

    # Reconstruction (on top of original)
    if plot_reconstruction:
        ax2 = fig.add_subplot(gs[6:8],sharex=ax0,sharey=ax0)
        ax2.plot(x,np.sum(filtered_df,axis=0),linestyle='dotted',alpha=0.9,label='filterbank reconstruction')
        ax2.legend(loc='upper right',bbox_to_anchor=(1.1, 1.2),fontsize=8)
        last_gs = 9
    else:
        last_gs = 6
    
    # Filtered Signal Decomposition
    filtered_df = fa.get_filtered_signals(data=data,
                                       fb_matrix=fb_matrix,
                                       fftfreq=fftfreq,
                                       cadence=cadence)
    for i,bank in enumerate(filtered_df):
        ax3 = fig.add_subplot(gs[last_gs+2*i:last_gs+2*i+2],sharex=ax0)    
        ax3.plot(x,bank)
        if center_freq is not None:
            ax3.text(x=min(x),y=max(bank),s=f'center freq = {center_freq[i]:.2e}',
                 ha='left',va='top',
                 fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        if i != filtered_df.shape[0]-1:
            ax3.tick_params(labelbottom=False)
        ax3.grid(True)
        if i==0:
            ax3.set_title('Signal decomposition',fontsize=15)
    if sig_xlim is not None:
        ax3.set_xlim(sig_xlim)

        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    # Synoptic map
    ax4 = fig.add_subplot(gs[-5:])
    ax4.imshow(syn_map_data[::-1,::-1],cmap=syn_map_cmap,
               extent = [-0.5,syn_map_data.shape[1]-0.5,-89.5,89.5],
               aspect='auto')
    ax4.set_xticklabels([])
    # ax4.set_xticklabels(pd.date_range(sig_xlim[0],sig_xlim[1],syn_map_data.shape[0]).strftime('%m-%d'))
    # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.show()


def filter_decomposition_2params(data,
                            fb_matrix,
                            fftfreq,
                            cadence = dt.timedelta(seconds=300),
                            figsize=(4,11),
                            gs_wspace = 0.2,
                            gs_hspace = 0.0,
                            fb_xlim = None,
                            sig_xlim = None,
                            center_freq = None,
                            filterbank_plot_title='Filter bank',
                            fb_freq_units = '',
                            fb_log_freq = False,
                            fb_plot_sci_not = True,
                            plot_reconstruction=False,
                            ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    assert len(data.columns) == 2, "data df needs to have exactly two columns"

    x = data.index
    y = data

    gs_recon = 3 if plot_reconstruction else 0
    # initialize gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = 3+3+gs_recon+fb_matrix.shape[0]*2,
                           figure = fig,
                           wspace=gs_wspace, 
                           hspace=gs_hspace)
    
    # Original series
    ax0 = fig.add_subplot(gs[0:2])   
    ax0.plot(data,color='black',label='original')
    ax0.set_ylabel('(nT)')
    ax0.set_title(f'Original Series [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}]')
    ax0.grid(True)

    # Filterbank plot
    
    ax1 = fig.add_subplot(gs[3:4])  
    if fb_xlim is not None:
        ax1.set_xlim(fb_xlim)
    ax1.plot(fftfreq, fb_matrix.T)
    ax1.grid(True)
    if fb_log_freq:
        ax1.set_xscale('log')
        fb_freq_units += ' [log scaled]'
    ax1.set_xlabel('Frequency'+fb_freq_units)
    
    ax1.set_title(filterbank_plot_title)
    # ax1.set_xticks(center_freq)
    ax1.tick_params(rotation=35,labelsize=8,axis='x')
    if fb_plot_sci_not:
        ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
    
    # Get filtered signals
    filtered = {}
    for col in data.columns:
        filtered[col] = fa.get_filtered_signals(data=data[col],
                                       fb_matrix=fb_matrix,
                                       fftfreq=fftfreq,
                                       cadence=cadence)
        

    # Reconstruction (on top of original)
    if plot_reconstruction:
        ax2 = fig.add_subplot(gs[6:8],sharex=ax0,sharey=ax0)
        for col in data.columns:
            ax2.plot(x,np.sum(filtered[col],axis=0),linestyle='dotted',alpha=0.9,label='filterbank reconstruction')
        ax2.legend(loc='upper right',bbox_to_anchor=(1.1, 1.2),fontsize=8)
        last_gs = 9
    else:
        last_gs = 6

    # Filtered Signal Decomposition
    for i in range(fb_matrix.shape[0]):
        ax3 = fig.add_subplot(gs[last_gs+2*i:last_gs+2*i+2],sharex=ax0)  
        ax31 = ax3.twinx()
        
        
        col1 = data.columns[0]
        color1 = 'tab:blue'
        ax3.plot(x,filtered[col1][i],
                 alpha=0.5,
                 color=color1,
                 label=col1)
        ax3.tick_params(axis='y',labelcolor=color1)

        col2 = data.columns[1]
        color2 = 'tab:orange'
        ax31.plot(x,filtered[col2][i],
                  alpha=0.5,
                  color=color2,
                  label=col2)
        ax31.tick_params(axis='y',labelcolor=color2)
            
        if center_freq is not None:
            ax3.text(x=min(x),y=max(filtered[col1][i]),s=f'center freq = {center_freq[i]:.2e}',
                ha='left',va='top',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        if i != fb_matrix.shape[0]-1:
            ax3.tick_params(labelbottom=False)
        
        # ax3.legend()
        # ax31.legend()
        ax3.grid(True)
        if i==0:
            ax3.set_title('Signal decomposition',fontsize=15)
    if sig_xlim is not None:
        ax3.set_xlim(sig_xlim)

        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.show()
    
def filter_decomposition_2params_with_synoptic(data,
                                       fb_matrix,
                                       fftfreq,
                                       syn_map_data,
                                       cadence = dt.timedelta(seconds=300),
                                       figsize=(4,11),
                                       gs_wspace = 0.2,
                                       gs_hspace = 0.0,
                                       fb_xlim = None,
                                       sig_xlim = None,
                                       center_freq = None,
                                       filterbank_plot_title='Filter bank',
                                       fb_freq_units = '',
                                       fb_log_freq = False,
                                       fb_plot_sci_not = True,
                                    #    orig_sig_plot_title='Original Signal',
                                       plot_reconstruction=False,
                                       syn_map_cmap = 'gray'
                                       ):
    """Plot comprehensive visualization of filterbank and its application to a set of test data.
    Plot includes the filterbank, raw test data, decomposition of filterbank preprocessed data.
    
    Parameters
    ----------
    
    """
    assert len(data.columns) == 2, "data df needs to have exactly two columns"
    x = data.index
    y = data

    orig_sig_plot_title = f'Original Series [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}]'
    if sig_xlim is not None:
        orig_sig_plot_title = f'Original Series [{sig_xlim[0].strftime('%Y-%m-%d')} to {sig_xlim[-1].strftime('%Y-%m-%d')}]'

    gs_recon = 3 if plot_reconstruction else 0
    # initialize gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = 3+3+gs_recon+fb_matrix.shape[0]*2+5, # original series + filterbank plot + reconstruction plot (if applicable) + decomposition
                           figure = fig,
                           wspace=gs_wspace, 
                           hspace=gs_hspace)
    
    # Original series
    ax0 = fig.add_subplot(gs[0:2])   
    ax0.plot(data,color='black',label='original')
    ax0.set_ylabel('(nT)')
    ax0.set_title(orig_sig_plot_title)
    ax0.grid(True)

    # Filterbank plot
    if fb_xlim is not None:
        ax1.set_xlim(fb_xlim)
    ax1 = fig.add_subplot(gs[3:4])  
    ax1.plot(fftfreq, fb_matrix.T)
    ax1.grid(True)
    if fb_log_freq:
        ax1.set_xscale('log')
        fb_freq_units += ' [log scaled]'
    ax1.set_xlabel('Frequency'+fb_freq_units)
    
    ax1.set_title(filterbank_plot_title)
    # ax1.set_xticks(center_freq)
    if fb_plot_sci_not:
        ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
        ax1.tick_params(rotation=35,labelsize=8,axis='x')

    # Get filtered signals
    filtered = {}
    for col in data.columns:
        filtered[col] = fa.get_filtered_signals(data=data[col],
                                       fb_matrix=fb_matrix,
                                       fftfreq=fftfreq,
                                       cadence=cadence)
        
     # Reconstruction (on top of original)
    if plot_reconstruction:
        ax2 = fig.add_subplot(gs[6:8],sharex=ax0,sharey=ax0)
        for col in data.columns:
            ax2.plot(x,np.sum(filtered[col],axis=0),linestyle='dotted',alpha=0.9,label='filterbank reconstruction')
        ax2.legend(loc='upper right',bbox_to_anchor=(1.1, 1.2),fontsize=8)
        last_gs = 9
    else:
        last_gs = 6

    # Filtered Signal Decomposition
    for i in range(fb_matrix.shape[0]):
        ax3 = fig.add_subplot(gs[last_gs+2*i:last_gs+2*i+2],sharex=ax0)  
        ax31 = ax3.twinx()
        
        
        col1 = data.columns[0]
        color1 = 'tab:blue'
        ax3.plot(x,filtered[col1][i],
                 alpha=0.5,
                 color=color1,
                 label=col1)
        ax3.tick_params(axis='y',labelcolor=color1)

        col2 = data.columns[1]
        color2 = 'tab:orange'
        ax31.plot(x,filtered[col2][i],
                  alpha=0.5,
                  color=color2,
                  label=col2)
        ax31.tick_params(axis='y',labelcolor=color2)
            
        if center_freq is not None:
            ax3.text(x=min(x),y=max(filtered[col1][i]),s=f'center freq = {center_freq[i]:.2e}',
                ha='left',va='top',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='black',alpha=0.7))
        if i != fb_matrix.shape[0]-1:
            ax3.tick_params(labelbottom=False)
        
        ax3.legend()
        ax3.grid(True)
        if i==0:
            ax3.set_title('Signal decomposition',fontsize=15)
    if sig_xlim is not None:
        ax3.set_xlim(sig_xlim)

        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    # Synoptic map
    ax4 = fig.add_subplot(gs[-5:])
    ax4.imshow(syn_map_data[::-1,::-1],cmap=syn_map_cmap,
               extent = [-0.5,syn_map_data.shape[1]-0.5,-89.5,89.5],
               aspect='auto')
    ax4.set_xticklabels([])
    # ax4.set_xticklabels(pd.date_range(sig_xlim[0],sig_xlim[1],syn_map_data.shape[0]).strftime('%m-%d'))
    # ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.show()