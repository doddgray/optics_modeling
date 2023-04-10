# -*- coding: utf-8 -*-
# Copyright 2019 Dodd Gray

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.patches as patches

from pathlib import Path
from typing import Union, Optional
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import  ListedColormap, LogNorm, Normalize, BoundaryNorm
from matplotlib.colors import  ListedColormap
from matplotlib import rcParamsDefault
from units_utils import u, Q_

## my default Parameters
default_plot_params = {
    'lines.linewidth': 1.5,
    'lines.markersize': 8,
    'legend.fontsize': 12,
    'text.usetex': False,
    # 'font.family': "serif",
    'font.serif': "cm",
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'font.size': 14,
    'axes.linewidth': 1,
    "grid.color": '#707070',
    'grid.linestyle':':',
    'grid.linewidth':0.7,
    'axes.grid': True,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'both',
    'image.cmap':'parula',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.dpi': 75,
    #'savefig.dpi': 75,
    'figure.autolayout': False,
    'figure.figsize': (10, 6),
}

## functions for twiny plotting (dual unit x axes sharing ticks)
def lm2f_tickfn(X,wl_units='nm',f_units='GHz',w=4,p=1):
    X_f = Q_(X,wl_units).to(f_units,'sp').m
    return [f'{z:{w}.{p}f}' for z in X_f]

def lm2f_tickfn_offset(X,offset_wl,wl_units='nm',f_units='GHz',w=4,p=1):
    # X_lm = Q_(X,wl_units)
    # offset_lm = Q_(offset,wl_units)
    X_f = Q_(X,wl_units).to(f_units,'sp').m  #(u.speed_of_light / X_lm).to(f_units).m
    offset_f = Q_(offset_wl,wl_units).to(f_units,'sp').m
    # offset_f_THz = (u.speed_of_light / offset_lm ).to(f_units).m
    X_offset = X_f - offset_f
    # return f"{offset_f_THz:{w}.{p}f}", [f"{z:{w}.{p}f}" for z in X_offset_GHz]
    return offset_f, [f'{z:{w}.{p}f}' for z in X_offset]

def f2lm_tickfn_offset(X,offset_f,wl_units='nm',f_units='GHz',w=4,p=1):
    X_wl = (Q_(X,f_units) + offset_f).to(wl_units,'sp').m  #(u.speed_of_light / X_lm).to(f_units).m
    offset_wl = Q_(offset_f,wl_units).to(f_units,'sp').m
    X_offset = X_wl - offset_wl
    return offset_wl, [f'{z:{w}.{p}f}' for z in X_offset]

def f2lm_tickfn(X,wl_units='nm',f_units='GHz',w=4,p=1):
    X_lm = Q_(X,f_units).to(wl_units,'sp').m
    return [f'{z:{w}.{p}f}' for z in X_lm]

def lm2f_twiny(ax,wl_units='nm',f_units='THz',w=4,p=1):
    ax2=ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(lm2f_tickfn(
        ticks,wl_units=wl_units,f_units=f_units,w=w,p=p))
    ax2.set_xlabel(f'frequency [{str(f_units)}]')
    return ax2

def lm2f_twiny_offset(ax,offset_wl=None,wl_units='nm',f_units='GHz',f_offset_units='THz',w=4,p=1):
    ax2=ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xbound(ax.get_xbound())
    if offset_wl is None:
        offset_wl = np.median(ticks)
    offset_freq, tick_str_list = lm2f_tickfn_offset(ticks,offset_wl,wl_units=wl_units,f_units=f_units,w=w,p=p)
    offset_str = f'{Q_(offset_freq,f_units).to(f_offset_units).m} {str(f_offset_units)}'
    ax2.set_xticklabels(tick_str_list)
    ax2.set_xlabel('frequency [' + str(f_units) + '] offset from ' + offset_str)
    return ax2

def f2lm_twiny_offset(ax,offset_freq=None,wl_units='pm',f_units='GHz',wl_offset_units='nm',w=4,p=1):
    ax2=ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xbound(ax.get_xbound())
    if offset_freq is None:
        offset_freq = np.median(ticks)
    offset_wl, tick_str_list = f2lm_tickfn_offset(ticks,offset_freq,wl_units=wl_units,f_units=f_units,w=w,p=p)
    offset_str = f'{Q_(offset_wl,wl_units).to(wl_offset_units).m} {str(wl_offset_units)}'
    # offset_str = f'{offset_wl:4.5g} {str(wl_offset_units)}'
    ax2.set_xticklabels(tick_str_list)
    ax2.set_xlabel('wavelength [' + str(wl_units) + '] offset from ' + offset_str)
    return ax2

def f2lm_twiny(ax,wl_units='nm',f_units='THz',w=4,p=1):
    ax2=ax.twiny()
    ticks = ax.get_xticks()
    ax2.set_xticks(ticks)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(f2lm_tickfn(ticks,wl_units=wl_units,f_units=f_units,w=w,p=p))
    ax2.set_xlabel(f'wavelength [{str(f_units)}]')
    return ax2

# add MATLAB's parula colormap
_parula_data = [[0.2081, 0.1663, 0.5292],
                [0.2116238095, 0.1897809524, 0.5776761905],
                [0.212252381, 0.2137714286, 0.6269714286],
                [0.2081, 0.2386, 0.6770857143],
                [0.1959047619, 0.2644571429, 0.7279],
                [0.1707285714, 0.2919380952, 0.779247619],
                [0.1252714286, 0.3242428571, 0.8302714286],
                [0.0591333333, 0.3598333333, 0.8683333333],
                [0.0116952381, 0.3875095238, 0.8819571429],
                [0.0059571429, 0.4086142857, 0.8828428571],
                [0.0165142857, 0.4266, 0.8786333333],
                [0.032852381, 0.4430428571, 0.8719571429],
                [0.0498142857, 0.4585714286, 0.8640571429],
                [0.0629333333, 0.4736904762, 0.8554380952],
                [0.0722666667, 0.4886666667, 0.8467],
                [0.0779428571, 0.5039857143, 0.8383714286],
                [0.079347619, 0.5200238095, 0.8311809524],
                [0.0749428571, 0.5375428571, 0.8262714286],
                [0.0640571429, 0.5569857143, 0.8239571429],
                [0.0487714286, 0.5772238095, 0.8228285714],
                [0.0343428571, 0.5965809524, 0.819852381],
                [0.0265, 0.6137, 0.8135],
                [0.0238904762, 0.6286619048, 0.8037619048],
                [0.0230904762, 0.6417857143, 0.7912666667],
                [0.0227714286, 0.6534857143, 0.7767571429],
                [0.0266619048, 0.6641952381, 0.7607190476],
                [0.0383714286, 0.6742714286, 0.743552381],
                [0.0589714286, 0.6837571429, 0.7253857143],
                [0.0843, 0.6928333333, 0.7061666667],
                [0.1132952381, 0.7015, 0.6858571429],
                [0.1452714286, 0.7097571429, 0.6646285714],
                [0.1801333333, 0.7176571429, 0.6424333333],
                [0.2178285714, 0.7250428571, 0.6192619048],
                [0.2586428571, 0.7317142857, 0.5954285714],
                [0.3021714286, 0.7376047619, 0.5711857143],
                [0.3481666667, 0.7424333333, 0.5472666667],
                [0.3952571429, 0.7459, 0.5244428571],
                [0.4420095238, 0.7480809524, 0.5033142857],
                [0.4871238095, 0.7490619048, 0.4839761905],
                [0.5300285714, 0.7491142857, 0.4661142857],
                [0.5708571429, 0.7485190476, 0.4493904762],
                [0.609852381, 0.7473142857, 0.4336857143],
                [0.6473, 0.7456, 0.4188],
                [0.6834190476, 0.7434761905, 0.4044333333],
                [0.7184095238, 0.7411333333, 0.3904761905],
                [0.7524857143, 0.7384, 0.3768142857],
                [0.7858428571, 0.7355666667, 0.3632714286],
                [0.8185047619, 0.7327333333, 0.3497904762],
                [0.8506571429, 0.7299, 0.3360285714],
                [0.8824333333, 0.7274333333, 0.3217],
                [0.9139333333, 0.7257857143, 0.3062761905],
                [0.9449571429, 0.7261142857, 0.2886428571],
                [0.9738952381, 0.7313952381, 0.266647619],
                [0.9937714286, 0.7454571429, 0.240347619],
                [0.9990428571, 0.7653142857, 0.2164142857],
                [0.9955333333, 0.7860571429, 0.196652381],
                [0.988, 0.8066, 0.1793666667],
                [0.9788571429, 0.8271428571, 0.1633142857],
                [0.9697, 0.8481380952, 0.147452381],
                [0.9625857143, 0.8705142857, 0.1309],
                [0.9588714286, 0.8949, 0.1132428571],
                [0.9598238095, 0.9218333333, 0.0948380952],
                [0.9661, 0.9514428571, 0.0755333333],
                [0.9763, 0.9831, 0.0538]]

parula = ListedColormap(_parula_data, name='parula')
plt.register_cmap(cmap=parula)

plt.rcParams.update(default_plot_params)


# from collections import defaultdict
# # define recursively nested default dict lambda fn
# nested_defaultdict = lambda: defaultdict(nested_defaultdict)
# textbox_params_default = nested_defaultdict()

def handle_axes(ax:Optional[plt.Axes]=None,**kwargs) -> plt.Axes:
    """If axes are provided (`ax`), just return those for plotting
    otherwise create a new figure and axes and return those. This 
    is a convenience method to allow plotting routines/recipes to
    work either as standalone plots or as subplots in a bigger 
    figure when a plt.Axes instance is passed."""
    if ax is None:
        fig,ax = plt.subplots(1,1,**kwargs)
    return ax

def add_textbox(
    ax:plt.Axes,
    textbox_str:str,
    x_textbox:float=0.02,         # X-position of text box in axes coords
    y_textbox:float=0.77,         # Y-position of text box in axes coords
    text_props = {
        'fontsize':7,
        'verticalalignment':'top',
        'bbox': {
            'facecolor':'0.8',
            'alpha':0.4,
            'boxstyle':'round',
        },
    },
):
    """Place a text box with multiline string `textbox_str` in axes coordinates."""
    ax.text(x_textbox,y_textbox,textbox_str,transform=ax.transAxes,**text_props)



"""
Saving and Exporting figures and other media.
"""



def save_figure(fpath,**kwargs):
    """Save a figure as both an SVG and PNG, and save
    PNG versions with both white and transparent backgrounds.
    This function lightly wraps `matplotlib.pyplot.savefig`"""
    fpath    =   Path(fpath)
    fstem    =   fpath.stem      #   remove filetype suffix if present
    save_dir =   fpath.parent    #   save directory
    # kwargs.update(transparent=True,facecolor=None,edgecolor=None)
    plt.savefig(save_dir.joinpath(fstem+'.svg'),transparent=True,**kwargs)
    plt.savefig(save_dir.joinpath(fstem+'.png'),transparent=True,**kwargs)
    # kwargs.update(transparent=False,facecolor='w',edgecolor='w')
    plt.savefig(save_dir.joinpath(fstem+'_whitebg.png'),transparent=False,facecolor='w',edgecolor='w',**kwargs)
