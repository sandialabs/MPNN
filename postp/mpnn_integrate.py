#!/usr/bin/env python

import sys
import time
import copy
import torch
import numpy as np


from mpnn.utils_mpnn import kB, trans_kinetic_factor, rot_kinetic_factor
from mpnn.utils_gen import myrc, loadpk
from mpnn.utils_int import IntegratorGMMT, IntegratorScipy, IntegratorMC, IntegratorMCMC

myrc()



####################################################################################
int_method = 'GMMT'
ads = 'CO' # This is only implemented for CO
####################################################################################

mpnn = loadpk('mpnn')
ncl = len(mpnn.mpts)
#print(mpts)
dim = mpnn.mpts[0].center.shape[0]


if ads == 'CO':
    r=0.680986 # for CO only!
    um = 12.0107 + 15.9994 # for CO only!
    momIn = 1.61795134850566E-46  # kg m2 (moment of inertia) # !!! is this CO specific?
else:
    print(f"Integration routine not ready for adsorbate {ads}. Exiting.")
    sys.exit()
#r=1
#um = 12.0107 + 1.00794*3 + 15.9994 + 1.00794 # CH3OH
#um = 1.00794 # H


TT=float(sys.argv[1])
nmc = int(sys.argv[2])
if len(sys.argv)>3:
    seed = int(sys.argv[3])
else:
    seed = None


def mpnneval_wrapper(x, mpnn=None,  temp=None):
    xr = x.copy()
    r=0.680986
    xr[:,:3] = mpnn.rhombi[mpnn.eval_type].fromCube(x[:, :3])
    xr[:, 4] = np.arccos(xr[:, 4])-np.pi/2.
    #xr[xr[:, 4]>np.pi/2., 4] -= np.pi
    xr[:, 4] *= r
    xr[:, 3] *= r
    y = mpnn.eval(xr, temp=temp)
    return y


# Integrate the MPNN wrapper function
func = mpnneval_wrapper
func_args = {'mpnn': mpnn, 'temp': TT}


volume_factor = np.linalg.det(mpnn.rhombi[mpnn.eval_type].transform) #/r**2
mpts_ = mpnn.rhombi[mpnn.eval_type].mpts_toCube(mpnn.mpts, xyfold=True)

domain = np.tile(np.array([0.0,1.0]), (dim,1))

if ads == 'CO':
    domain[3, :] = -np.pi, np.pi # for CO only!
    domain[4, :] = -1.0, 1.0 # -np.pi*r/2., np.pi*r/2. # for CO only!
else:
    print(f"Domain setup routine not ready for adsorbate {ads}. Exiting.")
    sys.exit()

## Truncated Gaussian Mixture Model Importance Sampling
if int_method == 'GMMT':

    # cosine transformation adjustment. Assumes the last parameter is theta, and minima are all at theta=0
    def coshess(hess):
        new_hess = hess.copy()
        new_hess[:,-1] *= -1
        new_hess[-1,:] *= -1
        return new_hess

    intg = IntegratorGMMT()
    integral, results = intg.integrate(func, domain,
                                      func_args=func_args,
                                      means=[mpt.center for mpt in mpts_],
                                      covs=[4.*np.linalg.inv(coshess(mpt.hess))*(kB*TT) for mpt in mpts_],
                                      nmc=nmc)
## Monte-Carlo integration, not accurate for low T
elif int_method == 'MC':
    intg = IntegratorMC(seed=seed)
    integral, results = intg.integrate(func,
                                     domain=domain,
                                     nmc=nmc, func_args=func_args)
else:
    print(f"Integration method {int_method} is unknown. Exiting.")
    sys.exit()


## Scipy integrator, goes on forever
# intg = IntegratorScipy()
# integral, results = intg.integrate(func, domain, func_args=func_args, epsrel=1.e-5)

## MCMC integration, not accurate
# domain = np.tile(np.array([-np.inf,np.inf]), (dim,1))
# intg = IntegratorMCMC()
# integral, results = intg.integrate(func,
#                                   func_args=func_args,
#                                   domain=domain,
#                                   nmc=nmc)


# Write out the samples for sanity check
if 'xdata' in results.keys():
    xr = results['xdata'].copy()
    xr[:,:3] = mpnn.rhombi[mpnn.eval_type].fromCube(results['xdata'][:, :3])
    np.savetxt('xdata_int.txt', xr)
if 'ydata' in results.keys():
    np.savetxt('ydata_int.txt', results['ydata'])


tf = trans_kinetic_factor(TT, um)
rf = rot_kinetic_factor(TT, um, momIn, ads='CO')

partition_function = tf*rf*volume_factor*integral
print(TT, nmc,  partition_function)
