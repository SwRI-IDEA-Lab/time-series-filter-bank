import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from astropy.io import fits
from sunpy.visualization.colormaps.color_tables import aia_color_table
import astropy.units as u

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
                         syn_map_data: np.ndarray = None,
                         cadence=dt.timedelta(seconds=300),
                         figsize=(4, 11),
                         gs_wspace=0.2,
                         gs_hspace=0.0,
                         plot_filterbank=True,
                         fb_xlim=None,
                         sig_xlim=None,
                         date_formatter = "%m-%d",
                         rotate_xticks = 0,
                         y_labels:list=None,
                         colors:list=None,
                         HR_ylim = None,
                         HRp1_ylim:bool=False,
                         center_freq=None,
                         filterbank_plot_title="Filter bank",
                         add_to_sig_title="",
                         fb_freq_units="",
                         fb_log_freq=False,
                         fb_plot_sci_not=True,
                         plot_reconstruction=False,
                         syn_map_wavelen: int = 193,
                         ):
    x = data.index
    orig_sig_plot_title = f"Original Signal [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"
    if sig_xlim is not None:
        orig_sig_plot_title = f"Original Signal [{sig_xlim[0].strftime('%Y-%m-%d')} to {sig_xlim[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"

    # Gridspec setup
    gs_fb = 3 if plot_filterbank else 0
    gs_recon = 3 if plot_reconstruction else 0
    gs_syn = 5 if syn_map_data is not None else 0
    total_gs_rows = 3 + gs_fb + gs_recon + fb_matrix.shape[0]*2 + gs_syn
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = total_gs_rows, 
                           figure = fig, 
                           wspace = gs_wspace, 
                           hspace = gs_hspace)

    # Plot original signal
    ax0 = fig.add_subplot(gs[0:2])
    if data.shape[1] == 1:
        col = data.columns[0]
        ax0.plot(data[col], color="black", label="original")
        ax0.set_ylabel(y_labels[0] if y_labels else col)
    elif data.shape[1] == 2:
        col1, col2 = data.columns
        ax01 = ax0.twinx()
        if colors is not None:
            c1 = colors[0]
            c2 = colors[1]
        else:
            c1 = 'blue'
            c2 = 'red'
        ax0.plot(data[col1], alpha=0.9, color=c1)
        ax01.plot(data[col2], alpha=0.7, color=c2)
        ax0.set_ylabel(y_labels[0] if y_labels else col1, color=c1)
        ax0.tick_params(axis="y", labelcolor=c1)
        ax01.set_ylabel(y_labels[1] if y_labels else col2, color=c2)
        ax01.tick_params(axis="y", labelcolor=c2)
    else:
        for col in data.columns:
            ax0.plot(data[col],alpha=0.75,label=col)
        ax0.legend()
        ax0.set_ylabel(y_labels[0])
    ax0.set_title(orig_sig_plot_title)
    ax0.grid(True)
    last_gs = 3

    # Filterbank plot
    if plot_filterbank:
        ax1 = fig.add_subplot(gs[last_gs : last_gs + 1])
        ax1.plot(fftfreq, fb_matrix.T)
        if fb_xlim:
            ax1.set_xlim(fb_xlim)
        if fb_log_freq:
            ax1.set_xscale("log")
            fb_freq_units += " [log scaled]"
        ax1.set_xlabel("Frequency" + fb_freq_units)
        ax1.set_title(filterbank_plot_title)
        ax1.grid(True)
        if fb_plot_sci_not:
            ax1.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
            ax1.tick_params(rotation=35, labelsize=8, axis="x")
        last_gs += 3

    # Filter
    filtered = {}
    for col in data.columns:
        filtered[col] = fa.get_filtered_signals(data=data[col], fb_matrix=fb_matrix, fftfreq=fftfreq, cadence=cadence)

    # Reconstruction plot
    if plot_reconstruction:
        ax2 = fig.add_subplot(gs[last_gs : last_gs + 2], sharex=ax0)
        for col in data.columns:
            ax2.plot(x, np.sum(filtered[col], axis=0), linestyle="dotted", alpha=0.9, label=f"{col} reconstruction")
        ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1.2), fontsize=8)
        last_gs += 3

    # Decomposition
    if data.shape[1]==2:
        c1 = f'xkcd:{c1}'
        c2 = f'xkcd:{c2}'
    for i in range(fb_matrix.shape[0]):
        ax = fig.add_subplot(gs[last_gs + 2 * i : last_gs + 2 * i + 2], sharex=ax0)
        if data.shape[1] == 1:
            col = data.columns[0]
            ax.plot(x, filtered[col][i])
            if i == fb_matrix.shape[0]//2:
                ax.set_ylabel(y_labels[0] if y_labels else col)
            
        elif data.shape[1] == 2:
            col1, col2 = data.columns
            ax_twin = ax.twinx()
            
            ax.plot(x, filtered[col1][i], color=c1, alpha=0.9)
            ax_twin.plot(x, filtered[col2][i], color=c2, alpha=0.75)
            if i == fb_matrix.shape[0] // 2:
                ax.set_ylabel(y_labels[0] if y_labels else col1, color=c1)
                ax_twin.set_ylabel(y_labels[1] if y_labels else col2, color=c2)
            ax.tick_params(axis="y", labelcolor=c1)
            ax_twin.tick_params(axis="y", labelcolor=c2)
            col = col1 # for the textbox
            if HR_ylim is not None:
                if (HRp1_ylim and i==fb_matrix.shape[0]-2) or i == fb_matrix.shape[0]-1:
                    lim1 = HR_ylim[0]
                    lim2 = HR_ylim[1]
                    ax.set_ylim(lim1[0],lim1[1])
                    ax_twin.set_ylim(lim2[0],lim2[1])

        else:
            for col in data.columns:
                ax.plot(x,filtered[col][i],label=col,alpha=0.75)
            if i == 0:
                ax.legend()
            if i == fb_matrix.shape[0]//2:
                ax.set_ylabel(y_labels[0])
            if HR_ylim is not None:
                if i == fb_matrix.shape[0]-1:
                    ax.margins(y=HR_ylim,tight=True)


        if center_freq is not None:
                ax.text(x=min(x), y=max(filtered[col][i]), s=f"center freq = {center_freq[i]:.2e}",
                        ha="left", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7))
        ax.grid(True)
        if i == 0:
            ax.set_title("Signal decomposition " + add_to_sig_title, fontsize=15)
        if i != fb_matrix.shape[0] - 1:
            ax.tick_params(labelbottom=False)

    last_gs += 2 * fb_matrix.shape[0] + 1
    if sig_xlim:
        ax.set_xlim(sig_xlim)
        ax0.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        ax0.set_xticklabels(ax0.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')

    if syn_map_data is not None:
        ax4 = fig.add_subplot(gs[-5:])
        ax4.imshow(
            syn_map_data[::-1, ::-1],
            cmap=aia_color_table(syn_map_wavelen * u.angstrom),
            extent=[-0.5, syn_map_data.shape[1] - 0.5, -89.5, 89.5],
            aspect="auto",
        )
        ax4.set_xticklabels([])
    plt.tight_layout()
    plt.show()

def stack_subdecomp(data,
                    main_filter,
                    sub_filter,
                    fftfreq,
                    cadence=dt.timedelta(seconds=300),
                    figsize=(4, 11),
                    gs_wspace=0.2,
                    gs_hspace=0.0,
                    main_alpha = 0.5,
                    sub_alpha = 0.7,
                    sig_xlim=None,
                    date_formatter = "%m-%d",
                    rotate_xticks = 0,
                    y_labels:list=None,
                    colors:list=None,
                    ylims = None,
                    add_to_sig_title=""):
    x = data.index
    orig_sig_plot_title = f"Original Signal [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"
    if sig_xlim is not None:
        orig_sig_plot_title = f"Original Signal [{sig_xlim[0].strftime('%Y-%m-%d')} to {sig_xlim[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"
    
    assert len(data.columns) % 2 ==0, "Even number of columns needed (Note: you can have the same data column twice to plot just one parameter in a single plot)"

    # Gridspec setup
    total_gs_rows =  len(data.columns) + 1 + len(data.columns)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = total_gs_rows, 
                           figure = fig, 
                           wspace = gs_wspace, 
                           hspace = gs_hspace)
    
    # Other prep
    col1s = data.columns[::2]
    col2s = data.columns[1::2]

    col_pairs = []
    for i,col1 in enumerate(col1s):
        col_pairs.append((col1,col2s[i]))

    if colors is None:
        colors = [('blue','red')]*len(col1s)

    # Plot original signal
    last_gs = 0
    for i in range(len(col1s)):
        ax0 = fig.add_subplot(gs[last_gs:last_gs+2])
        ax01 = ax0.twinx()
        y_lab = y_labels[i] if y_labels is not None else (col1s[i],col2s[i])
        clr = colors[i]

        ax0.plot(data[col1s[i]], alpha=0.9, color=clr[0])
        ax01.plot(data[col2s[i]], alpha=0.7, color=clr[1])
        ax0.set_ylabel(y_lab[0], color=clr[0])
        ax0.tick_params(axis="y", labelcolor=clr[0])
        ax01.set_ylabel(y_lab[1], color=clr[1])
        ax01.tick_params(axis="y", labelcolor=clr[1])
        
        if i == 0:
            ax0.set_title(orig_sig_plot_title)
        
        if sig_xlim:
            ax0.set_xlim(sig_xlim)
            ax0.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
            ax0.set_xticklabels(ax0.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')

        if i != len(col1s)-1:
            ax0.tick_params(labelbottom=False)

        ax0.grid(True)
        last_gs+=2

    

    # Filtered signals
    filtered_main = {}
    for col in np.unique(data.columns):
        if sum(data.columns == col)>1:
            d_col = data[col].iloc[:,0]
        else:
            d_col = data[col]
        filtered_main[col] = fa.get_filtered_signals(data=d_col, fb_matrix=np.array([main_filter]), fftfreq=fftfreq, cadence=cadence)

    filtered_sub = {}
    for col in np.unique(data.columns):
        if sum(data.columns == col)>1:
            d_col = data[col].iloc[:,0]
        else:
            d_col = data[col]
        filtered_sub[col] = fa.get_filtered_signals(data=d_col, fb_matrix=np.array([sub_filter]), fftfreq=fftfreq, cadence=cadence)

    # Filtered signal plots
    f_colors = []
    for cp in colors:
        clr1 = cp[0]
        clr2 = cp[1]
        f_colors.append((f'xkcd:{clr1}',f'xkcd:{clr2}'))

    last_gs +=1
    for i in range(len(col1s)):
        ax2 = fig.add_subplot(gs[last_gs : last_gs + 2], sharex=ax0)
        ax21 = ax2.twinx()

        f_clr = f_colors[i]
        y_lab = y_labels[i] if y_labels is not None else (col1s[i],col2s[i])
        
        sublinewidth = 2
        main_alpha = main_alpha
        sub_alpha = sub_alpha
        ax2.plot(x, filtered_main[col1s[i]][0], color=f_clr[0], alpha=main_alpha)
        ax2.plot(x, filtered_sub[col1s[i]][0],color=f_clr[0], linewidth=sublinewidth, alpha=sub_alpha)
        ax21.plot(x, filtered_main[col2s[i]][0], color=f_clr[1], alpha=main_alpha)
        ax21.plot(x, filtered_sub [col2s[i]][0], color=f_clr[1],linewidth=sublinewidth, alpha=sub_alpha)
        
        ax2.set_ylabel(y_lab[0], color=f_clr[0])
        ax2.tick_params(axis='y',labelcolor=f_clr[0])
        ax21.set_ylabel(y_lab[1], color=f_clr[1])
        ax21.tick_params(axis='y',labelcolor=f_clr[1])

        if ylims is not None:
            assert len(ylims)==len(col1s), "Not enough ylims provided for each pair of data"
            lims = ylims[i]
            lim1 = lims[0]
            lim2 = lims[1]
            ax2.set_ylim(lim1[0],lim1[1])
            ax21.set_ylim(lim2[0],lim2[1])

        if i == 0:
            ax2.set_title("Filtered Signals " + add_to_sig_title, fontsize=15)

        if i != len(col1s)-1:
            ax2.tick_params(labelbottom=False)
        
        ax2.grid(True)

        last_gs+=2
    
    if sig_xlim:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        ax2.set_xticklabels(ax2.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')

    
    plt.tight_layout()
    plt.show()