#!/bin/bash -e

export MPNN=$MPNN
#$WW/repos/MPNN

$MPNN/prep/mpnn_load.py 8
$MPNN/prep/prep_xytrain.x 8
$MPNN/prep/plot_xdata.py
$MPNN/prep/plot_ydata.py
$MPNN/prep/plot_yx.py
$MPNN/prep/plot_rhombus.py


$MPNN/fit/mpnn_fit.py 

$MPNN/postp/plot_zslice.py 
$MPNN/postp/plot_2dslice_xtrain.py 
$MPNN/postp/plot_2dslice_grid.py 
$MPNN/postp/plot_2dslice_tile.py 
$MPNN/postp/plot_datacl.py 


