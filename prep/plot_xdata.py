#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_gen import loadpk, myrc
from mpnn.utils_mpnn import plot_xdata
myrc()

case_glob = loadpk('case_glob')

pnames = case_glob['pnames']
mpts = case_glob['mpts']

xtrain = np.loadtxt('xtrain.txt')

plot_xdata(xtrain, mpts=mpts, pnames=pnames, every=1)

print([mpt.center for mpt in mpts])
