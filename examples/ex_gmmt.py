#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_int import IntegratorGMMT, IntegratorMC


def doublewell(x):
    return np.sum(x**2, axis=1)*np.sum((x-2.)**2, axis=1)

def expfunc(x, beta=1.0, Vfunc=None, thermo='I0'):

    if Vfunc is None:
        Vfunc = lambda x: np.sum(x**2, axis=1)/2.

    y = np.exp(-beta*Vfunc(x))

    if thermo == 'I1':
        y *= beta * Vfunc(x)
    elif thermo == 'I2':
        y *= (beta * Vfunc(x))**2


    return y

################################################

# Define dimensionaliy
dim = 3


# Define the integrand and its parameters and domain
# Note that this particular integrand is of form e^(-beta*V) for a given V
myfunc = expfunc
beta = 0.1
func_args = {'beta': beta, 'Vfunc': doublewell, 'thermo': 'I0'}
domain = np.tile(np.array([-4.0,14.0]), (dim,1))
# Minima of V are assumed known: list of minima/Hessian pairs
minima = [(np.zeros(dim), np.eye(dim)), (2.*np.ones(dim), np.eye(dim))]


# GMMT Integrator
# Note that for best results GMMT needs to use minima locations and inverse Hessians as covariance
intg = IntegratorGMMT()
int_gmmt, results = intg.integrate(myfunc, domain,
                                  func_args=func_args,
                                  means=[m[0] for m in minima],
                                  covs=[np.linalg.inv(m[1])/beta for m in minima],
                                  nmc=10000)
print(f"GMMT Estimate : {int_gmmt}")

# MC Integrator
intg = IntegratorMC()
int_mc, results = intg.integrate(myfunc,
                                 domain=domain,
                                 nmc=10000000, func_args=func_args)
print(f"MC Estimate   : {int_mc}")


