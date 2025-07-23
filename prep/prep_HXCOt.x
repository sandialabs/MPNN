#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${WW}/projects/ECC/run/surr_int/HXCO/TS_data_try1

echo "Loading training data for HXCO-t"

cp $DIR/s_train.dat xtrain.txt


cp $DIR/y_train.dat ytrain.txt


awk '{print "pt"}'  ytrain.txt > ltrain.txt


## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
awk '$6>0.0 && $6<4.0{print}' xyl > xyl_
awk '{print $1, $2, $3, $4, $5}' xyl_ > xtrain.txt
awk '{print $6}' xyl_ > ytrain.txt
awk '{for(i=7;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

##################################################################
##################################################################
#rm xyl xyl_