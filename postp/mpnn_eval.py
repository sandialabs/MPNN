#!/usr/bin/env python

import sys
import time
import torch
import argparse
import numpy as np

import pickle as pk



####################################################################################
####################################################################################

usage_str = 'Script to build PC surrogates of multioutput models.'
parser = argparse.ArgumentParser(description=usage_str)
parser.add_argument("-m", "--mpnn", dest="mpnn_pk", type=str, default='mpnn.pk',
                    help="Pk file of trained MPNN")
parser.add_argument("-x", "--xinput", dest="xinput", type=str, default='xtrain.txt',
                    help="Input file")
args = parser.parse_args()




mpnn = pk.load(open(args.mpnn_pk, 'rb'))
#pk.dump(mpnn, open('mpnn.pk', 'wb'), 2)

xsamples = np.loadtxt(args.xinput)
print(f"Evaluating at {xsamples.shape[0]} points")


# Evaluate the surrogate
start = time.time()
ysamples = mpnn.eval(xsamples, eps=0.02)
end = time.time()
print("Time :", end - start)

# Save
np.savetxt('mpnn_'+args.xinput, ysamples)

