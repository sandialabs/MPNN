MPNN is a small library for Minima-preserving neural network construction. See 
Katrín Blöndal, Khachik Sargsyan, David H. Bross, Branko Ruscic, and C. Franklin Goldsmith, "Adsorbate Partition Functions via Phase Space Integration: Quantifying the Effect of Translational Anharmonicity on Thermodynamic Properties", The Journal of Physical Chemistry C 2021 125 (37), 20249-20260. DOI: 10.1021/acs.jpcc.1c04009.

# Build the library
./build.sh

# Load data
	prep/prep_xytrain.x <caseid>
	prep/mpnn_load.py <caseid>
where <caseid> = 1 for H PBE-D3(ABC), 2 for H BEEF_vdW, 3 for CO, 4 for CH3OH, 5 for CH3OH_2d


# Exploratory data analysis
prep/plot_yx.py 
prep/plot_xdata.py 
prep/plot_ydata.py

# Training
mpnn/mp_fit.py

# Postprocess
postp/plot_parity.py
postp/plot_datacl.py
postp/plot_zslice.py
postp/plot_2dslice_tile.py
postp/plot_2dslice_xtrain.py
postp/plot_2dslice_grid.py
postp/mpnn_eval.py
postp/mpnn_integrate.py <T> <N> # only for CO for now