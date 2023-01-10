#!/usr/bin/env python
"""
Main code for MPNN fit
"""
import numpy as np

from mpnn.mpnn import MPNN
from mpnn.mpnn_plain import MPNN_plain
from mpnn.utils_gen import loadpk, savepk


case_glob = loadpk('case_glob')


xall = np.loadtxt('xtrain.txt')
if len(xall.shape)==1: xall = xall[:, np.newaxis]
yall = np.loadtxt('ytrain.txt')
npt, ndim =  xall.shape

mpnn = MPNN(case_glob)
mpnn.fit(xall, yall,
         tr_frac=0.9, cl_rad=111.5,
         hls=(111, 111, 111), activ='sigm',
         lr=0.01, nepochs=2000, bsize=2000, periodic_lambda=0.0,
         eps=0.02, eval_type=0)

# mpnn = MPNN_plain(case_glob)
# mpnn.fit(xall, yall,
#          tr_frac=0.9, cl_rad=2.5,
#          hls=(111, 111, 111), activ='sigm',
#          lr=0.01, nepochs=2000, bsize=2000,
#          eps=0.02)


savepk(mpnn, 'mpnn')
