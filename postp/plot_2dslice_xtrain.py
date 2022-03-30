#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt


from mpnn.utils_gen import loadpk, myrc, plot_2d_tri


myrc()
####################################################################################
####################################################################################
####################################################################################


mpnn = loadpk('mpnn')

ncl = len(mpnn.mpts)


xtrn = np.loadtxt('xtrain.txt')
#ytrn_ = np.loadtxt('ytrain.txt')


npt = xtrn.shape[0]

for icl in range(ncl):
    zval = mpnn.mpts[icl].center[2:]
    xyz = np.hstack((xtrn[:, :2], zval * np.ones((npt, zval.shape[0]))))

    ytrn = mpnn.eval(xyz, eps=0.02)


    ax = plot_2d_tri(xtrn[:, :2], ytrn, nlev=33)
    ax.plot(mpnn.mpts[icl].center[0], mpnn.mpts[icl].center[1], 'ro', markersize=16)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'slice_2d_{icl}_xtrain.png')
    plt.clf()
