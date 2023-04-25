#!/usr/bin/env python
"""Module for various plotting functions."""

import os
import sys
import warnings
import numpy as np

import colorsys
import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt



#############################################################

warnings.simplefilter(action='ignore', category=FutureWarning)

#############################################################

def myrc():
    """Configure matplotlib common look and feel.

    Returns:
        dict: Dictionary of matplotlib parameter config, not really used
    """
    mpl.rc('legend', loc='best', fontsize=22)
    mpl.rc('lines', linewidth=4, color='r')
    mpl.rc('axes', linewidth=3, grid=True, labelsize=22)
    mpl.rc('xtick', labelsize=15)
    mpl.rc('ytick', labelsize=15)
    mpl.rc('font', size=20)
    mpl.rc('figure', figsize=(12, 9), max_open_warning=200)
    #mpl.rc('lines', markeredgecolor='w')
    # mpl.rc('font', family='serif')

    return mpl.rcParams



#############################################################

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Args:
        color (str or tuple): Initial color.
        amount (float): How much to lighten: should be between 0 and 1.

    Returns:
        str: lightened color.

    Examples:
        >>> lighten_color('g', 0.3)
        >>> lighten_color('#F034A3', 0.6)
        >>> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = np.array(colorsys.rgb_to_hls(*mc.to_rgb(c)))
    return colorsys.hls_to_rgb(c[0],1-amount * (1-c[1]),c[2])


#############################################################
