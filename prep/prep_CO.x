#!/bin/bash -e

THISDIR=$(dirname "$0")

export DIR=${THISDIR}/../data/CO
#export DIR=${WW}/projects/ECC/run/surr_int/CO_Pt111/same_r_CO_Pt111_coords
##export DIR=${WW}/projects/ECC/run/surr_int/CO_Pt111/CO_Pt111_new_training_data
##export DIR=${WW}/projects/ECC/run/surr_int/h_3d/CO_Pt111_DFT_training

echo "Loading training data for CO"


#awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_fcc_hcp_atop.txt > xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_fcc_Hessian.txt > xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_fcc_MVN.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_hcp_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_hcp_MVN.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_atop_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_atop_MVN.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_Sobol.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/bridge_sites/xtrain_bridge_leftright_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/bridge_sites/xtrain_bridge_middle_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/bridge_sites/xtrain_bridge_lowerupper_Hessian.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/bridge_sites/xtrain_lower_middle_left.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/bridge_sites/xtrain_bridge_MVN.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/scaled_MVN/xtrain_fcc_scaledMVN0005.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/scaled_MVN/xtrain_fcc_scaledMVN001.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/scaled_MVN/xtrain_hcp_scaledMVN0005.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/scaled_MVN/xtrain_hcp_scaledMVN001.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_fcc_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_hcp_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_atop_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_bridge_leftright_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_bridge_middle_random.txt >> xtrain.txt
awk '{print $1, $2, $3, $4, $5}' $DIR/random_testing/xtest_bridge_lowerupper_random.txt >> xtrain.txt


#awk '{print $1}' $DIR/ytrain_fcc_hcp_atop.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_fcc_Hessian.txt > ytrain.txt
awk '{print $1}' $DIR/ytrain_fcc_MVN.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_hcp_MVN.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_atop_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_atop_MVN.txt >> ytrain.txt
awk '{print $1}' $DIR/ytrain_Sobol.txt >> ytrain.txt
awk '{print $1}' $DIR/bridge_sites/ytrain_bridge_leftright_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/bridge_sites/ytrain_bridge_middle_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/bridge_sites/ytrain_bridge_lowerupper_Hessian.txt >> ytrain.txt
awk '{print $1}' $DIR/bridge_sites/ytrain_lower_middle_left.txt >> ytrain.txt
awk '{print $1}' $DIR/bridge_sites/ytrain_bridge_MVN.txt >> ytrain.txt
awk '{print $1}' $DIR/scaled_MVN/ytrain_fcc_scaledMVN0005.txt >> ytrain.txt
awk '{print $1}' $DIR/scaled_MVN/ytrain_fcc_scaledMVN001.txt >> ytrain.txt
awk '{print $1}' $DIR/scaled_MVN/ytrain_hcp_scaledMVN0005.txt >> ytrain.txt
awk '{print $1}' $DIR/scaled_MVN/ytrain_hcp_scaledMVN001.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_fcc_random.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_hcp_random.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_atop_random.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_bridge_leftright_random.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_bridge_middle_random.txt >> ytrain.txt
awk '{print $1}' $DIR/random_testing/ytest_bridge_lowerupper_random.txt >> ytrain.txt



#awk '{print "minima"}' $DIR/ytrain_fcc_hcp_atop.txt > ytrain.txt
awk '{print "fcc Hessian"}' $DIR/ytrain_fcc_Hessian.txt > ltrain.txt
awk '{print "fcc MVN"}' $DIR/ytrain_fcc_MVN.txt >> ltrain.txt
awk '{print "hcp Hessian"}' $DIR/ytrain_hcp_Hessian.txt >> ltrain.txt
awk '{print "hcp MVN"}' $DIR/ytrain_hcp_MVN.txt >> ltrain.txt
awk '{print "atop Hessian"}' $DIR/ytrain_atop_Hessian.txt >> ltrain.txt
awk '{print "atop MVN"}' $DIR/ytrain_atop_MVN.txt >> ltrain.txt
awk '{print "Sobol"}' $DIR/ytrain_Sobol.txt >> ltrain.txt
awk '{print "bridge leftright Hessian"}' $DIR/bridge_sites/ytrain_bridge_leftright_Hessian.txt >> ltrain.txt
awk '{print "bridge middle Hessian"}' $DIR/bridge_sites/ytrain_bridge_middle_Hessian.txt >> ltrain.txt
awk '{print "bridge lowerupper Hessian"}' $DIR/bridge_sites/ytrain_bridge_lowerupper_Hessian.txt >> ltrain.txt
awk '{print "bridge lower middle left"}' $DIR/bridge_sites/ytrain_lower_middle_left.txt >> ltrain.txt
awk '{print "bridge MVN"}' $DIR/bridge_sites/ytrain_bridge_MVN.txt >> ltrain.txt
awk '{print "fcc scaledMVN0005"}' $DIR/scaled_MVN/ytrain_fcc_scaledMVN0005.txt >> ltrain.txt
awk '{print "fcc scaledMVN001"}' $DIR/scaled_MVN/ytrain_fcc_scaledMVN001.txt >> ltrain.txt
awk '{print "hcp scaledMVN0005"}' $DIR/scaled_MVN/ytrain_hcp_scaledMVN0005.txt >> ltrain.txt
awk '{print "hcp scaledMVN001"}' $DIR/scaled_MVN/ytrain_hcp_scaledMVN001.txt >> ltrain.txt
awk '{print "fcc randomtst"}' $DIR/random_testing/ytest_fcc_random.txt >> ltrain.txt
awk '{print "hcp randomtst"}' $DIR/random_testing/ytest_hcp_random.txt >> ltrain.txt
awk '{print "atop randomtst"}' $DIR/random_testing/ytest_atop_random.txt >> ltrain.txt
awk '{print "bridge leftright randomtst"}' $DIR/random_testing/ytest_bridge_leftright_random.txt >> ltrain.txt
awk '{print "bridge middle randomtst"}' $DIR/random_testing/ytest_bridge_middle_random.txt >> ltrain.txt
awk '{print "bridge lowerupper randomtst"}' $DIR/random_testing/ytest_bridge_lowerupper_random.txt >> ltrain.txt

## Problem specific down-selection
paste xtrain.txt ytrain.txt ltrain.txt > xyl
awk '$6<10 && $3<3.5{print}' xyl > xyl_
awk '{print $1, $2, $3, $4, $5}' xyl_ > xtrain.txt
awk '{print $6}' xyl_ > ytrain.txt
awk '{for(i=7;i<=NF;++i)printf("%s ", $i); print("")}' xyl_ > ltrain.txt

##################################################################
##################################################################

# awk '{print $1, $2, $3, $4, $5}' $DIR/xtrain_sobol_old.txt > xtst.txt
# awk '{print $1}' $DIR/ytrain_sobol_old.txt > ytst.txt

# paste xtst.txt ytst.txt > xy
# awk '$6<10 && $3<3.5{print}' xy > xy_
# awk '{print $1, $2, $3, $4, $5}' xy_ > xtst.txt
# awk '{print $6}' xy_ > ytst.txt


#rm xyl xyl_
