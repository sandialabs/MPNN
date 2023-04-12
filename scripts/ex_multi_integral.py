#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from mpnn.utils_int import IntegratorGMMT, IntegratorMC


def doublewell(x):
    print(f"Expensive! Number of evaluations {x.shape[0]}")
    return np.sum(x**2, axis=1)*np.sum((x-2.)**2, axis=1)

def expfunc(x, beta=1.0, Vfunc=None, thermo='I0', saved=None):

    if Vfunc is None:
        assert(saved is not None)
        Vfuncx = saved
    else:
        Vfuncx = Vfunc(x)

    y = np.exp(-beta*Vfuncx)

    if thermo == 'I1':
        y *= beta * Vfuncx
    elif thermo == 'I2':
        y *= (beta * Vfuncx)**2


    return y, Vfuncx

################################################

# Define dimensionaliy
dim = 3


# Define the list of (function, argument) pairs
beta = 0.1
domain = np.tile(np.array([-4.0,14.0]), (dim,1))

func_args1 = {'beta': beta, 'Vfunc': doublewell, 'thermo': 'I0'}
func_args2 = {'beta': beta, 'Vfunc': None, 'thermo': 'I1'}
func_args3 = {'beta': beta, 'Vfunc': None, 'thermo': 'I2'}

f_arg_list = [(expfunc,func_args1), (expfunc,func_args2), (expfunc,func_args3)]

# Minima of V are assumed known: list of minima/Hessian pairs
# Note: GMMT actually requires the V-values of the minima as well, but getting those is cheap enough, so they are evaluated within the integration routine.
minima = [(np.zeros(dim), np.eye(dim)), (2.*np.ones(dim), np.eye(dim))]


# GMMT Integrator
# Note that for best results GMMT needs to use minima locations and inverse Hessians as covariance
intg = IntegratorGMMT()
int_gmmt = intg.integrate_multiple(f_arg_list, domain=domain,
                                  means=[m[0] for m in minima],
                                  covs=[np.linalg.inv(m[1])/beta for m in minima],
                                  nmc=10000)
print(f"GMMT Estimates (I0, I1, I2) : {int_gmmt}")

# MC Integrator
intg = IntegratorMC()
int_mc = intg.integrate_multiple(f_arg_list, domain=domain, nmc=10000000)
print(f"MC Estimates  (I0, I1, I2) : {int_mc}")


