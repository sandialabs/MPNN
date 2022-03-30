#!/usr/bin/env python

import sys
import time
import torch
import numpy as np

import pickle as pk



####################################################################################
####################################################################################

if len(sys.argv) == 1:
    mpnn_pk_name = 'mpnn.pk'
elif len(sys.argv) == 2:
    mpnn_pk_name = sys.argv[1]
else:
    print("Please provide at most one argument.")
    sys.exit()

mpnn = pk.load(open(mpnn_pk_name, 'rb'))
#pk.dump(mpnn, open('mpnn.pk', 'wb'), 2)


xsamples = np.loadtxt('xtrain.txt')
print(f"Evaluating at {xsamples.shape[0]} points")


# Evaluate the surrogate
start = time.time()
ysamples = mpnn.eval(xsamples, eps=0.02)
end = time.time()
print("Time :", end - start)

# Save
np.savetxt('ytrain_surr.txt', ysamples)

