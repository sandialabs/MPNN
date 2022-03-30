#!/usr/bin/env python

import torch
import numpy as np

from mpnn.utils_mpnn import kB, plot_integrand_surr
from mpnn.utils_gen import loadpk, myrc

myrc()

####################################################################################
####################################################################################
####################################################################################


mpnn = loadpk('mpnn')

xall = np.loadtxt('xtrain.txt')
yall = np.loadtxt('ytrain.txt')

nall, dim = xall.shape
assert(yall.shape[0] == nall)


# Evaluate the surrogate
yall_pred = mpnn.eval(xall, eps=0.02)

np.savetxt('ytrain_surr.txt', yall_pred)

plot_integrand_surr(xall, yall, yall_pred, xall, yall, yall_pred, showtest=False)




