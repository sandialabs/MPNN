#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/H_pbe
#export DIR=${WW}/projects/ECC/run/surr_int/h_3d/H_Cu111_DFTtraining

echo "Loading training data for H PBE(ABC)"

awk '{print $2, $3, $1}' $DIR/xtrain_DFT_Hess.txt > xtrain.txt
awk '{print $2, $3, $1}' $DIR/xtrain_DFT_sobol.txt >> xtrain.txt
awk 'NR>1{print $2, $3, $1}' $DIR/xtrain_DFT_mins.txt >> xtrain.txt
awk '{print $2, $3, $1}' ${DIR}_lastbatch/xtrain_Hess_lastbatch.txt >> xtrain.txt

awk '{print $1}' $DIR/ytrain_DFT_Hess.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_DFT_sobol.txt >> ytrain.txt
awk 'NR>1{print $1}' $DIR/ytrain_DFT_mins.txt >> ytrain.txt
awk '{print $1}' ${DIR}_lastbatch/ytrain_DFT_Hess_lastbatch.txt >> ytrain.txt
