#!/usr/bin/env python

import torch
import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_gen import loadpk, myrc


myrc()
####################################################################################
####################################################################################
####################################################################################

mpnn = loadpk('mpnn')



ngr = 1000

zgrid = np.linspace(mpnn.zmin, mpnn.zmax, ngr)


ncl = len(mpnn.mpts)

for icl in range(ncl):
    z_val = mpnn.mpts[icl].center[2]

    xsamples = np.tile(mpnn.mpts[icl].center,(ngr,1))
    xsamples[:,2] = zgrid

    # Evaluate the surrogate
    ysamples = mpnn.eval(xsamples, eps=0.02)

    plt.plot(zgrid, ysamples)

    e_val = mpnn.eval(mpnn.mpts[icl].center.reshape(1,-1), eps=0.02)
    if e_val < 1.e-6:
        print("Offsetting to make 0 visible")
        shifted_center = mpnn.mpts[icl].center.copy()
        ind = np.argmin(np.abs(zgrid - shifted_center[2]))
        shifted_center[2] = zgrid[ind]
        e_val = mpnn.eval(shifted_center.reshape(1,-1), eps=0.02)

    plt.plot(z_val, e_val, 'ro', markersize=16)

    plt.xlabel('z')
    plt.ylabel('E')
    plt.savefig(f'zslice_{icl}.png')
    plt.yscale('log')
    plt.savefig(f'zslice_{icl}_log.png')
    plt.clf()
