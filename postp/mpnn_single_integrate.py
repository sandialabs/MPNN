#!/usr/bin/env python

import sys
import time
import copy
import torch
import argparse
import numpy as np


from mpnn.utils_mpnn import kB, trans_kinetic_factor, rot_kinetic_factor, thermo_integrand
from mpnn.utils_gen import myrc, loadpk
from mpnn.utils_int import IntegratorGMMT, IntegratorScipy, IntegratorMC, IntegratorMCMC

myrc()


ads = 'CH3' # This is only implemented for CO and CH3

####################################################################################

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--temp", dest="TT",
                    type=float, default=1000.0, help="Temperature")
parser.add_argument("-n", "--nmc", dest="nmc",
                    type=int, default=10000, help="Number of samples")
parser.add_argument("-m", "--method", dest="int_method",
                    type=str, default='MC', help="ytest file", choices=['GMMT', 'MC'])
parser.add_argument("-s", "--seed", dest="seed",
                    type=int, default=None, help="Optional seed (relevant for MC method)")
parser.add_argument("-f", "--cfactor", dest="cfactor",
                    type=float, default=4.0, help="Covariance factor (relevant for GMMT method)")
parser.add_argument("-i", "--intg", dest="intg_type",
                    type=str, default='I0', help="Which thermo integral to compute", choices=['I0', 'I1', 'I2'])
args = parser.parse_args()

TT = args.TT
nmc = args.nmc
int_method = args.int_method
seed = args.seed
cfactor = args.cfactor
intg_type = args.intg_type

####################################################################################

mpnn = loadpk('mpnn')
ncl = len(mpnn.mpts)
#print(mpts)
dim = mpnn.mpts[0].center.shape[0]


if ads == 'CO':
    r=0.680986 # for CO only!
    um = 12.0107 + 15.9994 # for CO only!
    momIn = 1.61795134850566E-46  # kg m2 (moment of inertia) # !!! is this CO specific?
elif ads =='CH3':
    r=1.0
    um = 12.0107 + 3*1.00784
    momIn = [3.40913416646E-47, 3.40987618499E-47, 5.36904980271E-47] # kg m2 (moments of inertia)
else:
    print(f"Integration routine not ready for adsorbate {ads}. Exiting.")
    sys.exit()



################################################
# Define the list of (function, argument) pairs
beta = 1.0/(kB*TT)

func_args = {'mpnn': mpnn, 'beta': beta, 'Vfunc': 'new', 'thermo': intg_type, 'r': r}


volume_factor = np.linalg.det(mpnn.rhombi[mpnn.eval_type].transform) #/r**2
mpts_ = mpnn.rhombi[mpnn.eval_type].mpts_toCube(mpnn.mpts, xyfold=True)

domain = np.tile(np.array([0.0,1.0]), (dim,1))

if ads == 'CO':
    domain[3, :] = -np.pi, np.pi # for CO only!
    domain[4, :] = -1.0, 1.0 # -np.pi*r/2., np.pi*r/2. # for CO only!
elif ads == 'CH3':
    domain[3, :] = -np.pi, np.pi
    domain[4, :] = -1.0, 1.0
    domain[5, :] = -np.pi, np.pi
else:
    print(f"Domain setup routine not ready for adsorbate {ads}. Exiting.")
    sys.exit()


if int_method == 'GMMT':
    # GMMT Integrator
    # Note that for best results GMMT needs to use minima locations and inverse Hessians as covariance
    def coshess(hess):
        new_hess = hess.copy()
        new_hess[:,4] *= -1
        new_hess[4,:] *= -1
        return new_hess

    means = [mpt.center for mpt in mpts_]
    covs = [cfactor*np.linalg.inv(coshess(mpt.hess))*(kB*TT) for mpt in mpts_]
    func_args_I0=copy.deepcopy(func_args)
    func_args_I0['thermo']='I0'

    weights = [thermo_integrand(mean.reshape(1,-1), **func_args_I0)[0][0] * np.sqrt(np.linalg.det(cov)) for mean, cov in zip(means, covs)]

    intg = IntegratorGMMT()
    intg_value, results = intg.integrate(thermo_integrand, domain=domain,
                                      func_args=func_args,
                                      means=means,
                                      covs=covs,
                                      weights=weights,
                                      nmc=nmc)


## Monte-Carlo integration, not accurate for low T
elif int_method == 'MC':
    intg = IntegratorMC(seed=seed)
    intg_value, results = intg.integrate(thermo_integrand,
                                     domain=domain,
                                     nmc=nmc, func_args=func_args)

else:
    print(f"Integration method {int_method} is unknown. Exiting.")
    sys.exit()


#print(f"{method} Estimate : {intg_value}")


# # Write out the samples for sanity check
# if 'xdata' in results.keys():
#     xr = results['xdata'].copy()
#     xr[:,:3] = mpnn.rhombi[mpnn.eval_type].fromCube(results['xdata'][:, :3])
#     np.savetxt('xdata_int.txt', xr)
# if 'ydata' in results.keys():
#     np.savetxt('ydata_int.txt', results['ydata'])


tf = trans_kinetic_factor(TT, um)
rf = rot_kinetic_factor(TT, um, momIn, ads=ads)

partition_function = tf*rf*volume_factor*intg_value

print(TT, nmc, intg_value, partition_function)
