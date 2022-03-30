#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/H_beef
#export DIR=${WW}/projects/ECC/run/surr_int/h_3d/H_Cu111_BEEF_training_9382


echo "Loading training data for H BEEF"

awk '{print $2, $3, $1}' $DIR/xtrain_fcc_4122.txt > xtrain.txt
awk '{print $2, $3, $1}' $DIR/xtrain_hcp_4232.txt >> xtrain.txt
awk 'NR>2{print $2, $3, $1}' $DIR/xtrain_DFT_fcc_hcp_ontop_bridge.txt >> xtrain.txt
awk '{print $2, $3, $1}' $DIR/xtrain_sobol.txt >> xtrain.txt



awk '{print $1}' $DIR/ytrain_DFT_hess_fcc_4122.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_DFT_hess_hcp_4232.txt >> ytrain.txt
awk 'NR>2{print $1}' $DIR/ytrain_DFT_fcc_hcp_ontop_bridge.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_DFT_sobol.txt >> ytrain.txt

