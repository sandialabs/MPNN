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



def mpnneval_wrapper(x, mpnn=None,  temp=None):
    xr = x.copy()
    xr[:,:3] = mpnn.rhombi[mpnn.eval_type].fromCube(x[:, :3])
    y = mpnn.eval(xr, temp=temp)
    return y


func = mpnneval_wrapper
func_args = {'mpnn': mpnn, 'temp': TT}


volume_factor = np.linalg.det(mpnn.rhombi[mpnn.eval_type].transform)/r**2
mpts_ = mpnn.rhombi[mpnn.eval_type].mpts_toCube(mpnn.mpts, xyfold=True)

# Integrate the MPNN wrapper function
intg = IntegratorGMMT()
domain = np.tile(np.array([0.0,1.0]), (dim,1))

if ads == 'CO':
    domain[3, :] = -np.pi*r, np.pi*r # for CO only!
    domain[4, :] = -np.pi*r/2., np.pi*r/2. # for CO only!
else:
    print(f"Domain setup routine not ready for adsorbate {ads}. Exiting.")
    sys.exit()

integral, results = intg.integrate(func, domain,
                                  func_args=func_args,
                                  means=[mpt.center for mpt in mpts_],
                                  covs=[np.linalg.inv(mpt.hess)*(kB*TT) for mpt in mpts_],
                                  nmc=nmc)

# Scipy integrator, goes on forever
# intg = IntegratorScipy()
# integral, results = intg.integrate(func, domain, func_args=func_args, epsrel=1.e-5)

# Monte-Carlo integration, not accurate
# intg = IntegratorMC()
# integral, results = intg.integrate(func,
#                                  domain=domain,
#                                  nmc=nmc, func_args=func_args)

# MCMC integration, not accurate
# domain = np.tile(np.array([-np.inf,np.inf]), (dim,1))
# intg = IntegratorMCMC()
# integral, results = intg.integrate(func,
#                                   func_args=func_args,
#                                   domain=domain,
#                                   nmc=nmc)

print(integral)


tf = trans_kinetic_factor(TT, um)
rf = rot_kinetic_factor(TT, um, momIn, ads='CO')
partition_function = tf*rf*volume_factor*integral
print(TT, nmc,  partition_function)
