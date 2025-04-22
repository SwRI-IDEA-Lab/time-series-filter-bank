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
                         y_labels=None,
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
        ax0.plot(data[col1], alpha=0.9, color="blue")
        ax01.plot(data[col2], alpha=0.7, color="red")
        ax0.set_ylabel(y_labels[0] if y_labels else col1, color="blue")
        ax0.tick_params(axis="y", labelcolor="blue")
        ax01.set_ylabel(y_labels[1] if y_labels else col2, color="red")
        ax01.tick_params(axis="y", labelcolor="red")

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
    for i in range(fb_matrix.shape[0]):
        ax = fig.add_subplot(gs[last_gs + 2 * i : last_gs + 2 * i + 2], sharex=ax0)
        if data.shape[1] == 1:
            col = data.columns[0]
            ax.plot(x, filtered[col][i])
            if center_freq is not None:
                ax.text(x=min(x), y=max(filtered[col][i]), s=f"center freq = {center_freq[i]:.2e}",
                        ha="left", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7))
        else:
            col1, col2 = data.columns
            ax_twin = ax.twinx()
            ax.plot(x, filtered[col1][i], color="blue", alpha=0.9)
            ax_twin.plot(x, filtered[col2][i], color="red", alpha=0.7)
            if i == fb_matrix.shape[0] // 2:
                ax.set_ylabel(y_labels[0] if y_labels else col1, color="blue")
                ax_twin.set_ylabel(y_labels[1] if y_labels else col2, color="red")
            ax.tick_params(axis="y", labelcolor="blue")
            ax_twin.tick_params(axis="y", labelcolor="red")
            if center_freq is not None:
                ax.text(x=min(x), y=max(filtered[col1][i]), s=f"center freq = {center_freq[i]:.2e}",
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
        ax0.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    if syn_map_data is not None:
        ax4 = fig.add_subplot(gs[-5:])
        ax4.imshow(
            syn_map_data[::-1, ::-1],
            cmap=aia_color_table(syn_map_wavelen * u.angstrom),
            extent=[-0.5, syn_map_data.shape[1] - 0.5, -89.5, 89.5],
            aspect="auto",
        )
        ax4.set_xticklabels([])

    plt.show()
