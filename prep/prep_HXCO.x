#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${WW}/projects/ECC/run/surr_int/HXCO/HXCO_AdTherm_data_try6

echo "Loading training data for HXCO"

awk '{print $1, $2, $3, $4, $5, $6}' $DIR/x_train.dat > xtrain.txt
cat $DIR/stencil_train/stencil_x_train_1Eneg2.dat >> xtrain.txt
cat $DIR/xtrain_zslice.dat >> xtrain.txt
# awk 'NR<=303{print $1, $2, $3, $4, $5, $6}' $DIR/x_train.dat > xtrain.txt
# head -n3600 $DIR/stencil_train/stencil_x_train_1Eneg5.dat >> xtrain.txt


awk '{print $1}' $DIR/y_train_ensemble/y_train_0.dat > ytrain.txt
cat $DIR/stencil_train/stencil_y_train_1Eneg2.dat >> ytrain.txt
cat $DIR/ytrain_zslice.dat >> ytrain.txt
# awk 'NR<=303{print $1}' $DIR/y_train_ensemble/y_train_0.dat > ytrain.txt
# head -3600 $DIR/stencil_train/stencil_y_train_1Eneg5.dat >> ytrain.txt

cp $DIR/tags.txt ltrain.txt
awk '{print "Stcl"}'  $DIR/stencil_train/stencil_y_train_1Eneg2.dat >> ltrain.txt
cat $DIR/ltrain_zslice.dat >> ltrain.txt
# head -n303 $DIR/tags.txt > ltrain.txt
# awk 'NR<=3600{print "Stcl"}'  $DIR/stencil_train/stencil_y_train_1Eneg5.dat >> ltrain.txt


## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
awk '$7<100.0{print}' xyl > xyl_
awk '{print $1, $2, $3, $4, $5, $6}' xyl_ > xtrain.txt
awk '{print $7}' xyl_ > ytrain.txt
awk '{for(i=8;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

##################################################################
##################################################################
#rm xyl xyl_