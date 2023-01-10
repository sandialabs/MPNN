#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/CH3

echo "Loading training data for CH3"

awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_fcc_MVN.txt > xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_fcc_scaledMVN001.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_fcc_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_hcp_MVN.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_hcp_scaledMVN001.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_hcp_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_sobol.txt >> xtrain.txt


awk '{print $1}' $DIR/ytrain_fcc_MVN.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_fcc_scaledMVN001.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_fcc_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_MVN.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_scaledMVN001.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_random.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_sobol.txt >> ytrain.txt


awk '{print "fcc MVN"}' $DIR/ytrain_fcc_MVN.txt > ltrain.txt
awk '{print "fcc scaledMVN001"}' $DIR/ytrain_fcc_scaledMVN001.txt >> ltrain.txt
awk '{print "fcc Hessian"}' $DIR/ytrain_fcc_Hessian.txt >> ltrain.txt
awk '{print "hcp MVN"}' $DIR/ytrain_hcp_MVN.txt >> ltrain.txt
awk '{print "hcp scaledMVN001"}' $DIR/ytrain_hcp_scaledMVN001.txt >> ltrain.txt
awk '{print "hcp Hessian"}' $DIR/ytrain_hcp_Hessian.txt >> ltrain.txt
awk '{print "random"}' $DIR/ytrain_random.txt >> ltrain.txt
awk '{print "sobol"}' $DIR/ytrain_sobol.txt >> ltrain.txt

## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
#awk '$7<3 && $3<4.5{print}' xyl > xyl_
awk '$3<3.0{print}' xyl > xyl_
awk '{print $1, $2, $3, $4, $5, $6}' xyl_ > xtrain.txt
awk '{print $7}' xyl_ > ytrain.txt
awk '{for(i=8;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

##################################################################
##################################################################
#rm xyl xyl_