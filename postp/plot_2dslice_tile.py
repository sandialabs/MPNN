#!/usr/bin/env python

import sys
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt


from mpnn.utils_mpnn import Rhombus
from mpnn.utils_gen import myrc, loadpk, plot_2d_tri, linear_transform, cartes_list

myrc()



####################################################################################
####################################################################################
####################################################################################


ngr = 333

xx = np.linspace(-2, 7, ngr)
yy = np.linspace(-2, 7, ngr)

xgrid = np.array(cartes_list([xx, yy]))

# #case_glob = {'mpts': mpts, 'zmin': zmin, 'zmax': zmax, 'delta_x': delta_x, 'pnames': pnames}
# case_glob = loadpk('case_glob')

mpnn = loadpk('mpnn')

init = np.array([0.,0., mpnn.zmin])
rhomb_orig = Rhombus(init=init, delta_x=mpnn.delta_x,
                     delta_z=mpnn.zmax-mpnn.zmin)




npt = xgrid.shape[0]

ncl = len(mpnn.mpts)

ngrb=111
dim = 3
border_cube1 = np.hstack((np.linspace(0.0, 1.0, ngrb).reshape(-1,1), np.zeros((ngrb, dim-1))))
border_cube2 = np.hstack((np.ones((ngrb, 1)), np.linspace(0.0, 1.0, ngrb).reshape(-1,1), np.zeros((ngrb, dim-2))))
border_cube3 = np.hstack((np.linspace(1.0, 0.0, ngrb).reshape(-1,1), np.ones((ngrb, 1)), np.zeros((ngrb, dim-2))))
border_cube4 = np.hstack((np.zeros((ngrb, 1)), np.linspace(1.0, 0.0, ngrb).reshape(-1,1), np.zeros((ngrb, dim-2))))

border_cube = np.vstack((border_cube1, border_cube2, border_cube3, border_cube4))

for icl in range(ncl):
    print(f"Plotting the surrogate at a 2d slice at minimum {icl}")
    zval = mpnn.mpts[icl].center[2:]
    xyz = np.hstack((xgrid, zval * np.ones((npt, zval.shape[0]))))

    ygrid = mpnn.eval(xyz, eps=0.02)

    ax = plot_2d_tri(xgrid, ygrid, nlev=33)
    ax.plot(mpnn.mpts[icl].center[0], mpnn.mpts[icl].center[1], 'ro', markersize=16)

    for jcl in range(ncl):
        border_rhombus = mpnn.rhombi[jcl].fromCube(border_cube)
        lw=1
        if jcl==icl: lw*=2
        plt.plot(border_rhombus[:,0], border_rhombus[:,1], 'r--', linewidth=lw)

    border_rhombus_orig = rhomb_orig.fromCube(border_cube)
    plt.plot(border_rhombus_orig[:,0], border_rhombus_orig[:,1], 'm-', linewidth=lw/2)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'slice_2d_{icl}_tile.png')
    plt.clf()



for icl in range(ncl):
    print(f"Plotting the weights at a 2d slice at minimum {icl}")

    zval = mpnn.mpts[icl].center[2:]
    xyz = np.hstack((xgrid, zval * np.ones((npt, zval.shape[0]))))

    ypad = 0.6
    fig, axes = plt.subplots(ncl, 2, figsize=(16,(4+ypad)*ncl),
                             gridspec_kw={'hspace': ypad, 'wspace': 0.3})



    for jclw in range(ncl):
        border_rhombus = mpnn.rhombi[jclw].fromCube(border_cube)

        if ncl>1:
            _ = plot_2d_tri(xgrid, mpnn.mmodel.wfcn(xyz)[:, jclw], nlev=33, ax=axes[jclw, 0])
        curax = axes[1] if ncl == 1 else axes[jclw, 1]
        _ = plot_2d_tri(xgrid, mpnn.mmodel.models[jclw](xyz), nlev=33, ax=curax)
        for k in range(2):
            curax = axes[k] if ncl == 1 else axes[jclw, k]
            curax.plot(mpnn.mpts[icl].center[0], mpnn.mpts[icl].center[1], 'ro', markersize=16)
            curax.plot(border_rhombus[:,0], border_rhombus[:,1], 'r--', linewidth=1)
            curax.set_xlabel('x')
            curax.set_ylabel('y')
    plt.savefig(f'slice_2d_{icl}_tile_w.png')
    plt.clf()
