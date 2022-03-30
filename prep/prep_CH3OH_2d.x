#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/CH3OH

echo "Loading training data for CH3OH 2d"

awk '{print $1, $2, $3}' $DIR/xtrain_2Dslice.txt > xtrain.txt

awk '{print $1}' $DIR/ytrain_2Dslice.txt > ytrain.txt

awk '{print "2d slice"}' $DIR/ytrain_2Dslice.txt > ltrain.txt

## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
awk '$4<4.5{print}' xyl > xyl_
awk '{print $1, $2, $3}' xyl_ > xtrain.txt
awk '{print $4}' xyl_ > ytrain.txt
awk '{for(i=5;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

##################################################################
##################################################################
rm xyl xyl_