# Morphology analysis and plotting
import sys, os, re, numpy as np, pandas as pd

import matplotlib
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
from matplotlib import collections as mcol
from collections import OrderedDict, namedtuple

from .helpers import * 

Point = namedtuple("Point", "x y")


# package for converting .hoc Neuron morphology to .SWC file format
#from hoc2swc import hoc2swc # use pip install hoc2swc

global_yz_mark = []

from Bio import Phylo

# text buffer with file-like API
# find the best implementation available on this platform
from io import StringIO

import plotly.graph_objs as go
import ipywidgets as ipw
from plotly.offline import iplot, plot, init_notebook_mode

from neuron import h as nrn_h, nrn
nrn_imported = True

from matplotlib.lines import Line2D


def plot_morphology(hoc_fpath, annotation_df = None, ax = None, median_lw = 1, legend_fontsize = 12,
    legend_ncol = 3, legend_zoffset = -50, figsize = (10,10), default_marker_size = 5,
    default_marker_color = (1,0,0,0.5), open_hoc_file = True, **kwargs):
    """
    Plot morphology from .hoc file

    Parameters
    ----------
    hoc_fpath : str
        NEURON .hoc morphology file.

    annotation_df : None, pandas.DataFrame
        Annotation information as dataframe with columns:
            'section' : str
                Section name, e.g. 'soma' or 'dend[10]'.

            'location' : float
                Normalized segment location within the section [0,1].

            'marker_color' : tuple, str, default (1,0,0,0.5)
                Segment marker color.
                If tuple:
                    - RGB (r,g,b)
                    - RGB+alpha (r,g,b,a)
                If str:
                    - Named color.
                    - Hex color
                Or any other valid matplotlib color specifier.

            'annotation_color' : tuple, str, default (1,0,0,0.75)
                Segment annotation color. Same description as for 'marker_color'.

            'marker_size' : float, default 5
                Annotation marker size in points.

            'annotation_text' : str
                Annotation label.

            'annotation_text_size' : float, default 12
                Annotation font size in points.

            'legend_text' : str
                Legend text for this marker.

    ax : None, matplotlib.axes._subplots.AxesSubplot
        Plotting axes with '3d' projection.

    median_lw : float
        Median line weight used for plotting segments. Adjusting this parameter scales the thickness of segments.

    legend_fontsize : float
        Legend font size in points.

    legend_ncol : int
        Number of legend columns.

    legend_zoffset : float
        Legend z-axis offset in data corrdinates. More negative values will shift the legend towards the top.

    default_marker_size : float
        Marker size in pt.

    default_marker_color : tuple, str, default (1,0,0,0.75)
        Segment annotation color. Same description as for 'marker_color'.

    open_hoc_file : bool
        If True, opens morphology again. Set to False if this is not needed and the same morphology can be used for different
        sequential plots.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        Plotting axes.
    """
    # clear hoc of all sections and mechanisms and load .hoc file
    if open_hoc_file:
        load_hoc_file(nrn_h, hoc_fpath)

    # shapeplot
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize = figsize, subplot_kw = {'projection': '3d'}, constrained_layout = True)
    else:
        fig = None

    ax.view_init(elev = 0, azim = 0)

    weighed_shapeplot(nrn_h, ax, median_lw = median_lw)

    # remove grid lines
    ax.grid(False)
    
    # remove panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # remove ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])

    # remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.invert_zaxis()

    section_lookup = {str(sec): sec for sec in nrn_h.allsec()}

    if annotation_df is not None:
        for _, loc in annotation_df.iterrows():
            add_annotation(h = nrn_h, sec = section_lookup[loc.section], locs = loc.location,
                annotation_text = "", # annotation not working properly # loc.annotation_text if 'annotation_text' in loc and not pd.isnull(loc.annotation_text) else "",
                ax = ax,
                annotation_color = loc.annotation_color if 'annotation_color' in loc and not pd.isnull(loc.annotation_color) else (1,0,0,0.75),
                annotation_text_size = loc.annotation_text_size if 'annotation_text_size' in loc and not pd.isnull(loc.annotation_text_size) else 12,
                marker_color = loc.marker_color if 'marker_color' in loc and not pd.isnull(loc.marker_color) else default_marker_color,
                markersize = loc.marker_size if 'marker_size' in loc and not pd.isnull(loc.marker_size) else default_marker_size,
                legend_text = loc.legend_text if 'legend_text' in loc and not pd.isnull(loc.legend_text) else ""
                , **kwargs)
    # add legend if there are any legend labels
    if 'legend_text' in annotation_df and not annotation_df['legend_text'].isnull().all():
        handles, labels = ax.get_legend_handles_labels()

        if 'marker_color' in annotation_df:
            marker_colors = [mc if not pd.isnull(mc) else default_marker_color for mc in annotation_df['marker_color']]
        else:
            marker_colors = [default_marker_color]*len(labels)
        by_label = OrderedDict(zip(zip(labels, marker_colors), handles))
        if by_label:
            # transform object corrdinates to projection coordinates
            f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
            labels, handles = zip(*sorted(zip([k[0] for k in by_label.keys()], by_label.values()), key = lambda t: t[0]))
            
            ax.legend(handles, labels, loc = 'lower left', fontsize = legend_fontsize,
                ncol = legend_ncol, bbox_to_anchor = f(ax.get_xlim()[0],ax.get_ylim()[0],ax.get_zlim()[1]+legend_zoffset), bbox_transform = ax.transData)

    return fig, ax


def plot_phylogram(hoc_fpath, annotation_df = None, ax = None, figsize = (10,10), lw = 0.35, trunk_lw = 0.7,
    plotter = 'pyplot', legend_fontsize = 12, legend_ncol = 3, default_marker_size = 5,
    default_marker_color = (1,0,0,0.5), open_hoc_file = True, add_roi_target_labels = False,
    roi_target_label_xy_offset = (20, 0),  **kwargs):
    """
    Plots a dendritic tree phylogram.

    Parameters
    ----------
    hoc_fpath : str
        NEURON .hoc morphology file.
    annotation_df : None, pandas.DataFrame

        Annotation information as dataframe with columns:
            'section' : str
                Section name, e.g. 'soma' or 'dend[10]'.

            'location' : float
                Normalized segment location within the section [0,1].

            'marker_color' : tuple, str, default (1,0,0,0.5)
                Segment marker color.
                If tuple:
                    - RGB (r,g,b)
                    - RGB+alpha (r,g,b,a)
                If str:
                    - Named color.
                    - Hex color
                Or any other valid matplotlib color specifier.

            'annotation_color' : tuple, str, default (1,0,0,0.75)
                Segment annotation color. Same description as for 'marker_color'.

            'marker_size' : float, default 5
                Annotation marker size in points.

            'annotation_text' : str
                Annotation label.

            'annotation_text_size' : float, default 12
                Annotation font size in points.

            'legend_text' : str
                Legend text for this marker.

    ax : matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis to use for plotting.

    figsize : tuple
        Figure width and height in inches.

    lw : float
        Line weight in points for non-trunk sections (or all sections if 'trunk' section list is not specified in NEURON).

    trunk_lw : float
        Line weight in points for trunk sections.

    plotter : str
        Choice of plotting engine. Choose between 'pyplot' and 'plotly'.

    legend_fontsize : float
        Legend font size in pt.

    legend_ncol : int
        Number of legend columns.

    open_hoc_file : bool
        If True, opens morphology again. Set to False if this is not needed and the same morphology can be used for different
        sequential plots.

    add_roi_target_labels : bool
        If True, adds ROI scan target labels next to the marked scan locations that have an associated legend label.

    roi_target_label_xy_offset : tuple
        X & Y axis offset in data coordinates between the marked location and ROI scan target label lower left corner.
    """
    lines, line_widths, arcs, arc_widths, adjusted_annotation_df = _get_circular_tree_data(hoc_fpath = hoc_fpath, annotation_df = annotation_df,
        plotter = plotter, default_marker_size = default_marker_size, default_marker_color = default_marker_color,
        open_hoc_file = open_hoc_file, trunk_linewidth = trunk_lw, nontrunk_linewidth = lw)

    if plotter == 'pyplot': 
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize = figsize)
        else:
            fig = None

        # radial lines
        lines_coord = [[(x[0][idx], x[1][idx]) for idx in range(len(x[0]))] for x in zip(lines.x,lines.y)]
        lc = mcol.LineCollection(lines_coord, color = 'k', linewidths = line_widths, zorder = 1,
            joinstyle = 'round', capstyle = 'round')
        ax.add_collection(lc)

        # arcs
        arcs_coord = [[(x[0][idx], x[1][idx]) for idx in range(len(x[0]))] for x in zip(arcs.x,arcs.y)]
        lc = mcol.LineCollection(arcs_coord, color = (0.7, 0.7, 0.7), linewidths = arc_widths, zorder = 0,
            joinstyle = 'round', capstyle = 'round')
        ax.add_collection(lc)

        unique_markings = set()
        if not adjusted_annotation_df.empty:
            for i, loc in adjusted_annotation_df.iterrows():
                # mark location
                if loc.legend_text and (loc.legend_text, loc.marker_color) in unique_markings:
                    ax.scatter(loc.x, loc.y, c = loc.marker_color, s = loc.marker_size, zorder = 2, **kwargs)
                else:
                    if loc.legend_text:
                        unique_markings.add((loc.legend_text, loc.marker_color)) 
                    ax.scatter(loc.x, loc.y, c = loc.marker_color, s = loc.marker_size, label = loc.legend_text, zorder = 2, **kwargs)
                # add roi target name next to the marked location
                if add_roi_target_labels:
                    if loc.legend_text:
                        ax.text(x = loc.x+roi_target_label_xy_offset[0], y = loc.y+roi_target_label_xy_offset[1], s = loc.annotation_text,
                            color = loc.marker_color, zorder = 2)
        
        # autoscale and equalize aspect ratio
        ax.autoscale()
        ax.set_aspect('equal')
        # remove spines and ticks
        ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
        ax.tick_params(axis = 'y', which = 'both', right = False, left = False, labelleft = False)
        for pos in ['right','top','bottom','left']:
            ax.spines[pos].set_visible(False)

        # add scalebar
        scalebar = AnchoredHScaleBar(size = 100, label = "100 $\mu$m", loc = 4, frameon = False, 
            pad = 0.6, sep = 4, linekw = dict(color = "k", linewidth = 1.5), extent = 0.012, label_fontsize = 7)
        ax.add_artist(scalebar)
        
        # add legend if there are any legend labels
        default_marker_color = (1,0,0,0.5)
        if annotation_df is not None and 'legend_text' in annotation_df and not annotation_df['legend_text'].isnull().all():
            # sort by legend labels
            handles, labels = sort_legend_labels(ax)
            if handles and labels:
                ax.legend(handles, labels, loc = 'lower left', bbox_to_anchor= (0.0, 1.01), fontsize = legend_fontsize, ncol = legend_ncol)

        return fig, ax

    if plotter == 'plotly':

        trace_radial_lines = dict(
            type = 'scatter',
            x = lines.x,
            y = lines.y,
            mode = 'lines',
            line = dict(color = 'rgb(0,0,0)', width = 1),
            hoverinfo = 'none'
        )

        trace_arcs = dict(
            type = 'scatter',
            x = arcs.x,
            y = arcs.y,
            mode = 'lines',
            line = dict(color='rgb(180,180,180)', width = 1, shape = 'spline'),
            hoverinfo = 'none'
        )

        if not adjusted_annotation_df.empty:
            trace_nodes = dict(
                type = 'scatter',
                x = adjusted_annotation_df.x,
                y = adjusted_annotation_df.y, 
                mode = 'markers',
                marker = dict(
                    color = adjusted_annotation_df.marker_color,
                    size = adjusted_annotation_df.marker_size,
                ),
                text = adjusted_annotation_df.segment, 
                hoverinfo = "text"
            )
        else:
            trace_nodes = {}

        layout = dict(
            font = dict(family = 'Balto', size = 14),
            width = 700,
            height = 750,
            autosize = False,
            showlegend = False,
            xaxis = dict(visible = False),
            yaxis = dict(visible = False, scaleanchor = "x", scaleratio = 1), 
            hovermode = 'closest',
            plot_bgcolor = 'rgb(245,245,245)',
            margin = dict(t = 75)
        )

        fig = go.FigureWidget(data = [trace_radial_lines, trace_arcs, trace_nodes], layout = layout)
        fig.update_layout(
            {
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            }
        )

        init_notebook_mode(connected = True)
        iplot(fig)

        return fig

    else:
        raise ValueError("plotter parameter must be 'pyplot' or 'plotly'.")
