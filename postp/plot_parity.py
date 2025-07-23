#!/usr/bin/env python

import torch
import argparse
import numpy as np
import pickle as pk

from mpnn.utils_mpnn import kB, plot_integrand_surr
from mpnn.utils_gen import loadpk, myrc

myrc()

####################################################################################
####################################################################################
####################################################################################

usage_str = 'Script for parity plots to check the accuracy of MPNN approximation.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-m", "--mpnn", dest="mpnn_pk", type=str, default='mpnn.pk',
                    help="Pk file of trained MPNN")
parser.add_argument("-x", "--xinput", dest="xinput", type=str, default='xtrain.txt',
                    help="Input file")
parser.add_argument("-y", "--youtput", dest="youtput", type=str, default='ytrain.txt',
                    help="Output file")
args = parser.parse_args()


mpnn = pk.load(open(args.mpnn_pk, 'rb'))

xall = np.loadtxt(args.xinput)
yall = np.loadtxt(args.youtput)

nall, dim = xall.shape
assert(yall.shape[0] == nall)


# Evaluate the surrogate
yall_pred = mpnn.eval(xall, eps=0.2)

np.savetxt('mpnn_'+args.youtput, yall_pred)

#plot_integrand_surr(xall, yall, yall_pred, xall, yall, yall_pred, showtest=False)
plot_integrand_surr(xall, yall, yall_pred, xall, yall, yall_pred, showtest=False, Trange=np.arange(300,901,300))




