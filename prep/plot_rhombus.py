#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from mpnn.mpnn import MPNN
from mpnn.utils_mpnn import plot_xdata
from mpnn.utils_gen import loadpk, myrc

myrc()

case_glob = loadpk('case_glob')

pnames = case_glob['pnames']
mpts = case_glob['mpts']

xtrain = np.loadtxt('xtrain.txt')

mpnn = MPNN(case_glob)
mpnn.set_rhombi()

ncl = len(mpnn.rhombi)

fig, axs = plt.subplots(ncl, 2, figsize=(15, 20), \
                        gridspec_kw={'width_ratios': [2, 1.5]}, constrained_layout=True)

axs = axs.reshape(-1,2)

for i in range(ncl):
    print(f"Rhombus {i}: {mpnn.rhombi[i].init=}")
    mpnn.rhombi[i].plotit(ax=axs[i,0])
    axs[i,0].plot(xtrain[:,0], xtrain[:,1], 'o', markeredgecolor='w')
    mpt = mpnn.mpts[i].center[:2]
    axs[i,0].plot(mpt[0], mpt[1], 'r*', markersize=11)
    xtrain_cube = mpnn.rhombi[i].toCube(xtrain[:,:3], xyfold=True)
    axs[i, 1].plot(xtrain_cube[:,0], xtrain_cube[:,1], 'o', markeredgecolor='w')
    axs[i, 1].plot([0., 0., 1., 1., 0.], [0., 1., 1., 0., 0.], 'k--')

plt.savefig('rhombi.png')


