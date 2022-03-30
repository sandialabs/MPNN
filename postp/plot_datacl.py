#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt


from mpnn.utils_gen import myrc, loadpk

myrc()
case_glob = loadpk('case_glob')

mpts = case_glob['mpts']
pnames = case_glob['pnames']
ncl = len(mpts)

colors = ['r', 'g', 'b']
styles = ['o', 's', 'd']

xall = np.loadtxt('xtrain.txt')
indtrn = np.loadtxt('indtrn.txt', dtype=int)
xtrain = xall[indtrn, :]

ndim = xtrain.shape[1]

for idim in range(ndim):
    for jdim in range(idim+1, ndim):
        for icl in range(ncl):
            indsel = np.loadtxt(f'indsel_{icl}_trn.txt', dtype=int)
            plt.plot(xtrain[indsel, idim], xtrain[indsel, jdim], colors[icl]+styles[icl], alpha=0.5, markeredgecolor='w')
            plt.plot(mpts[icl].center[idim], mpts[icl].center[jdim], colors[icl]+styles[icl], alpha=1.-icl/ncl, markersize=16, zorder=10000)

        plt.xlabel(f'{pnames[idim]}')
        plt.ylabel(f'{pnames[jdim]}')
        plt.savefig(f'xtrain_{idim}_{jdim}_cl.png')
        plt.clf()
