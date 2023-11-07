#!/bin/bash -e

export MPNN=${WW}/repos/MPNN
export MDIR=${WW}/projects/ECC/run/surr_int/mp_nn/run_CH3_conv

# Initially copy xtrain/ytrain to xall/yall

ln -sf ${MDIR}/case_glob.pk .
paste ${MDIR}/xall.txt ${MDIR}/yall.txt > xyall.txt

sort -R xyall.txt > xyall_shuffled.txt
head -n8000 xyall_shuffled.txt > xytrain.txt
tail -n861 xyall_shuffled.txt > xytest.txt

awk '{print $1, $2, $3, $4, $5, $6}' xytest.txt > xtest.txt
awk '{print $7}' xytest.txt > ytest.txt

for NN in 8000 4000 2000 1000 500 250 125; do
    echo "NN=${NN} ##################################################"
    sort -R xytrain.txt | head -n ${NN} > xytrain_this.txt
    awk '{print $1, $2, $3, $4, $5, $6}' xytrain_this.txt > xtrain.txt
    awk '{print $7}' xytrain_this.txt > ytrain.txt

    # train
    ${MPNN}/fit/mpnn_fit.py
    cp mpnn.pk mpnn_${NN}.pk
    # postprocess
    ${MPNN}/postp/plot_parity.py -m mpnn.pk -x xtest.txt -y ytest.txt > rrmse_test_N${NN}.txt 
done

