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


def stack_subdecomp(data1,
                    data2,
                    main_filter,
                    sub_filter,
                    fftfreq,
                    cadence=dt.timedelta(seconds=300),
                    figsize=(4, 11),
                    gs_wspace=0.2,
                    gs_hspace=0.0,
                    plot_filterbank=False,
                    sig_xlim=None,
                    date_formatter = "%m-%d",
                    rotate_xticks = 0,
                    y_labels1:list=None,
                    y_labels2:list=None,
                    colors1:list=None,
                    colors2:list=None,
                    ylim1 = None,
                    ylim2 = None,
                    add_to_sig_title=""):
    x = data1.index
    orig_sig_plot_title = f"Original Signal [{x[0].strftime('%Y-%m-%d')} to {x[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"
    if sig_xlim is not None:
        orig_sig_plot_title = f"Original Signal [{sig_xlim[0].strftime('%Y-%m-%d')} to {sig_xlim[-1].strftime('%Y-%m-%d')}] {add_to_sig_title}"
    
    # Gridspec setup
    gs_fb = 3 if plot_filterbank else 0
    total_gs_rows = 1 + 1 + gs_fb + 3 + 2 
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(ncols = 1, 
                           nrows = total_gs_rows, 
                           figure = fig, 
                           wspace = gs_wspace, 
                           hspace = gs_hspace)

    # Plot original signal
    ax0 = fig.add_subplot(gs[0:1])
    ax1 = fig.add_subplot(gs[1:2],sharex=ax0)

    col11, col12 = data1.columns
    ax01 = ax0.twinx()

    col21, col22 = data2.columns
    ax11 = ax1.twinx()

    if colors1 is not None:
        c11 = colors1[0]
        c12 = colors1[1]
    else:
        c11 = 'blue'
        c12 = 'red'
    if colors2 is not None:
        c21 = colors2[0]
        c22 = colors2[1]
    else:
        c21 = 'turquoise'
        c22 = 'orange'

    ax0.plot(data1[col11], alpha=0.9, color=c11)
    ax01.plot(data1[col12], alpha=0.7, color=c12)
    ax0.set_ylabel(y_labels1[0] if y_labels1 is not None else col11, color=c11)
    ax0.tick_params(axis="y", labelcolor=c11)
    ax01.set_ylabel(y_labels1[1] if y_labels1 is not None else col12, color=c12)
    ax01.tick_params(axis="y", labelcolor=c12)

    ax0.set_title(orig_sig_plot_title)
    ax0.grid(True)

    ax1.plot(data2[col21],alpha=0.9, color = c21)
    ax11.plot(data2[col22],alpha=0.7, color =c22)
    ax1.set_ylabel(y_labels2[0] if y_labels2 is not None else col21,color=c21)
    ax1.tick_params(axis="y", labelcolor=c21)
    ax11.set_ylabel(y_labels2[1] if y_labels2 is not None else col22, color=c22)
    ax11.tick_params(axis='y', labelcolor=c22)
    ax1.grid(True)

    ax0.tick_params(labelbottom=False)
    last_gs = 2

    # # Filterbank plot
    # if plot_filterbank:
    #     ax1 = fig.add_subplot(gs[last_gs : last_gs + 1])
    #     ax1.plot(fftfreq, fb_matrix.T)
    #     if fb_xlim:
    #         ax1.set_xlim(fb_xlim)
    #     if fb_log_freq:
    #         ax1.set_xscale("log")
    #         fb_freq_units += " [log scaled]"
    #     ax1.set_xlabel("Frequency" + fb_freq_units)
    #     ax1.set_title(filterbank_plot_title)
    #     ax1.grid(True)
    #     if fb_plot_sci_not:
    #         ax1.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")
    #         ax1.tick_params(rotation=35, labelsize=8, axis="x")
    #     last_gs += 3

    # Filtered signals
        #data 1
    filtered1_main = {}
    for col in data1.columns:
        filtered1_main[col] = fa.get_filtered_signals(data=data1[col], fb_matrix=np.array([main_filter]), fftfreq=fftfreq, cadence=cadence)

    filtered1_sub = {}
    for col in data1.columns:
        filtered1_sub[col] = fa.get_filtered_signals(data=data1[col], fb_matrix=np.array([sub_filter]), fftfreq=fftfreq, cadence=cadence)

        #data 2
    filtered2_main = {}
    for col in data2.columns:
        filtered2_main[col] = fa.get_filtered_signals(data=data2[col],fb_matrix=np.array([main_filter]),fftfreq=fftfreq,cadence=cadence)

    filtered2_sub = {}
    for col in data2.columns:
        filtered2_sub[col] = fa.get_filtered_signals(data=data2[col],fb_matrix=np.array([sub_filter]),fftfreq=fftfreq,cadence=cadence)

    # Filtered signal plots
    fc11 = f'xkcd:{c11}'
    fc12 = f'xkcd:{c12}'
    fc21 = f'xkcd:{c21}'
    fc22 = f'xkcd:{c22}'

    ax2 = fig.add_subplot(gs[last_gs + 1 : last_gs + 3], sharex=ax0)
    ax21 = ax2.twinx()

    ax3 = fig.add_subplot(gs[last_gs+3:last_gs+5],sharex=ax0)
    ax31 = ax3.twinx()
    
    sublinewidth = 2
    ax2.plot(x, filtered1_main[col11][0], color=fc11, alpha=0.5)
    ax2.plot(x,filtered1_sub[col11][0],color=c11, linewidth=sublinewidth, alpha=0.7)
    ax21.plot(x, filtered1_main[col12][0], color=fc12, alpha=0.5)
    ax21.plot(x, filtered1_sub [col12][0], color=fc12,linewidth=sublinewidth, alpha=0.7)
    
    ax2.set_ylabel(y_labels1[0] if y_labels1 is not None else col11, color=fc11)
    ax2.tick_params(axis='y',labelcolor=fc11)
    ax21.set_ylabel(y_labels1[1] if y_labels1 is not None else col12, color=fc12)
    ax21.tick_params(axis='y',labelcolor=fc12)

    ax3.plot(x,filtered2_main[col21][0],color=fc21, alpha=0.5)
    ax3.plot(x,filtered2_sub[col21][0],color=fc21,linewidth=sublinewidth,alpha=0.7)
    ax31.plot(x,filtered2_main[col22][0],color=fc22,alpha=0.5)
    ax31.plot(x,filtered2_sub[col22][0],color=fc22,linewidth=sublinewidth,alpha=0.7)

    ax3.set_ylabel(y_labels2[0] if y_labels2 is not None else col21, color=fc21)
    ax3.tick_params(axis='y',labelcolor=fc11)
    ax31.set_ylabel(y_labels2[1] if y_labels2 is not None else col22, color=fc22)
    ax31.tick_params(axis='y',labelcolor=fc22)

    if ylim1 is not None:
        lim11 = ylim1[0]
        lim12 = ylim1[1]
        ax2.set_ylim(lim11[0],lim11[1])
        ax21.set_ylim(lim12[0],lim12[1])
    if ylim2 is not None:
        lim21 = ylim2[0]
        lim22 = ylim2[1]
        ax3.set_ylim(lim21[0],lim21[1])
        ax31.set_ylim(lim22[0],lim22[1])

        ax2.grid(True)
        ax3.grid(True)
        
        ax2.set_title("Filtered Signal " + add_to_sig_title, fontsize=15)

        ax2.tick_params(labelbottom=False)

    if sig_xlim:
        ax0.set_xlim(sig_xlim)
        # ax1.set_xlim(sig_xlim)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        ax1.set_xticklabels(ax1.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')
        ax3.xaxis.set_major_formatter(mdates.DateFormatter(date_formatter))
        ax3.set_xticklabels(ax3.get_xticklabels(),rotation=rotate_xticks,ha='right',rotation_mode='anchor')

    
    plt.tight_layout()
    plt.show()