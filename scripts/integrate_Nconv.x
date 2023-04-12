#!/bin/bash -e

export MPNN=${WW}/repos/MPNN

for TH in I0 I1 I2; do
    for NN in 100 200 400 800 1600 3200 6400 12800 25600 51200 102400 1000 10000 100000; do 
        echo "$TH N=$NN ##############################"
        ${MPNN}/scripts/integrate_T.x $TH $NN GMMT
        ${MPNN}/scripts/integrate_T.x $TH $NN MC
    done
done