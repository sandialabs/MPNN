#!/usr/bin/env python
"""
Main code for MPNN fit
"""
import numpy as np

from mpnn.mpnn import MPNN
from mpnn.utils_gen import loadpk, savepk


case_glob = loadpk('case_glob')


xall = np.loadtxt('xtrain.txt')
yall = np.loadtxt('ytrain.txt')
npt, ndim =  xall.shape

mpnn = MPNN(case_glob)
mpnn.fit(xall, yall,
         tr_frac=0.9, cl_rad=111.5,
         hls=(111, 111, 111), activ='sigm',
         lr=0.01, nepochs=1000, bsize=500, periodic_lambda=0.0,
         eps=0.02, eval_type=0)

savepk(mpnn, 'mpnn')
