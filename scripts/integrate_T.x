#!/bin/bash -e

export MPNN=${WW}/repos/MPNN

thermo=$1 #'I0' #'I2' #'I1'
NN=$2 #Number of integration points
method=$3 #'GMMT' 'MC'

ww=16

filename="int_${method}_N${NN}_${thermo}.txt"

rm -rf $filename
echo -n''> $filename
#for T in $(seq 290 74 1400); do
#for T in $(seq 300 250 800); do
for T in $(seq 300 100 1400); do
    echo "T=$T ##############"
    for i in $(seq 1 1 50); do
        #echo "i=$i ######"
        ${MPNN}/postp/mpnn_single_integrate.py -t ${T} -n ${NN} -m ${method} -f ${ww} -i ${thermo} >> ${filename}
        #${MPNN}/postp/mpnn_multi_integrate.py -t ${T} -n ${NN} -m ${method} -f ${ww}  >> ${filename}
    done
done