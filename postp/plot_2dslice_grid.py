#!/usr/bin/env python

import sys
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt


from mpnn.utils_gen import myrc, loadpk, plot_2d_tri, linear_transform, cartes_list

myrc()



####################################################################################
####################################################################################
####################################################################################


ngr = 333

xx = np.linspace(0, 1, ngr)
yy = np.linspace(0, 1, ngr)
zz = np.array([0.0])

xgrid = np.array(cartes_list([xx, yy, zz]))


mpnn = loadpk('mpnn')



xgrid_rhomb = mpnn.rhomb_eval.fromCube(xgrid)

ncl = len(mpnn.mpts)



npt = xgrid.shape[0]


for icl in range(ncl):
    zval = mpnn.mpts[icl].center[2:]

    xyz = np.hstack((xgrid_rhomb[:, :2], zval * np.ones((npt, zval.shape[0]))))

    ygrid = mpnn.eval(xyz, eps=0.02)


    ax = plot_2d_tri(xgrid_rhomb[:, :2], ygrid,
                     nlev=33, cbar_lims=[0.0, 0.2])
    ax.plot(mpnn.mpts[icl].center[0], mpnn.mpts[icl].center[1], 'ro', markersize=16)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'slice_2d_{icl}_grid.png')
    plt.clf()


