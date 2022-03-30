#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_gen import plot_yx, read_textlist, loadpk, myrc
myrc()

case_glob = loadpk('case_glob')

pnames = case_glob['pnames']
xall = np.loadtxt('xtrain.txt')
yall = np.loadtxt('ytrain.txt')

every = 1
nrows = len(pnames)
plot_yx(xall[::every], yall[::every], rowcols=(nrows, 1), ylabel='E', xlabels=pnames, log=False, filename='e1d.png', ypad=1.2, gridshow=False, ms=4)
plot_yx(xall[::every], yall[::every], rowcols=(nrows, 1), ylabel='E', xlabels=pnames, log=True, filename='e1d_log.png', ypad=1.2, gridshow=False, ms=4)

