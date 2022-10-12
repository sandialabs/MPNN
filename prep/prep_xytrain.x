#!/bin/bash -e

CASEID=$1

THISDIR=$(dirname "$0")

if [[ $CASEID == 1 ]]; then
    ${THISDIR}/prep_H_pbe.x
elif [[ $CASEID == 2 ]]; then
    ${THISDIR}/prep_H_beef.x
elif [[ $CASEID == 3 ]]; then
    ${THISDIR}/prep_CO.x
elif [[ $CASEID == 4 ]]; then
    ${THISDIR}/prep_CH3OH.x
elif [[ $CASEID == 5 ]]; then
    ${THISDIR}/prep_CH3OH_2d.x
elif [[ $CASEID == 6 ]]; then
    ${THISDIR}/prep_CH3.x
else
    echo "Case Id unknown. Should be 1 for H pbe-d3(abc), 2 for H beef, 3 for CO, 4 for CH3OH, 5 for CH3OH_2d"
fi