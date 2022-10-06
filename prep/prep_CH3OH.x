#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/CH3OH
#export DIR=${WW}/projects/ECC/run/surr_int/CH3OH_Cu111/CH3OH_Cu111_6Dtraining

echo "Loading training data for CH3OH"

awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_atop_MVN.txt > xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_atop_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_sobol.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_scaledMVN0005.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtest_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtest_random_subspace.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_2Dslice.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_fcc_scaledMVN005.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_hcp_scaledMVN005.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5, $6}' $DIR/xtrain_bridge_sp_scaledMVN005.txt >> xtrain.txt

awk '{print $1}' $DIR/ytrain_atop_MVN.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_atop_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_sobol.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_scaledMVN0005.txt >> ytrain.txt
awk '{print $1}' $DIR/ytest_random.txt >> ytrain.txt
awk '{print $1}' $DIR/ytest_random_subspace.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_2Dslice.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_fcc_scaled_MVN005_500.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_scaled_MVN005_500.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_bridge_sp_scaled_MVN005_500.txt >> ytrain.txt

awk '{print "atop MVN"}' $DIR/ytrain_atop_MVN.txt > ltrain.txt
awk '{print "atop Hessian"}' $DIR/ytrain_atop_Hessian.txt >> ltrain.txt
awk '{print "Sobol"}' $DIR/ytrain_sobol.txt >> ltrain.txt
awk '{print "atop scaledMVN0005"}' $DIR/ytrain_scaledMVN0005.txt >> ltrain.txt
awk '{print "random test"}' $DIR/ytest_random.txt >> ltrain.txt
awk '{print "random subspace test"}' $DIR/ytest_random_subspace.txt >> ltrain.txt
awk '{print "2d slice"}' $DIR/ytrain_2Dslice.txt >> ltrain.txt
awk '{print "fcc hollow"}' $DIR/ytrain_fcc_scaled_MVN005_500.txt >> ltrain.txt
awk '{print "hcp hollow"}' $DIR/ytrain_hcp_scaled_MVN005_500.txt >> ltrain.txt
awk '{print "bridge saddle"}' $DIR/ytrain_bridge_sp_scaled_MVN005_500.txt >> ltrain.txt

## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
awk '$7<3 && $3<4.5{print}' xyl > xyl_
awk '{print $1, $2, $3, $4, $5, $6}' xyl_ > xtrain.txt
awk '{print $7}' xyl_ > ytrain.txt
awk '{for(i=8;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

#awk '{print $1-2.173805, $2-0.316083, $3, $4, $5, $6}' xtrain.txt > aa; mv aa xtrain.txt
##################################################################
##################################################################
#rm xyl xyl_