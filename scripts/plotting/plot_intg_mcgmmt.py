#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import myrc, lighten_color

myrc()

N_MC=10000
N_GMMT=10000
suffix = '_w4'
thermo='I2'

nrepl = 10

plt.figure(figsize=(12,10))

TNIQ1 = np.loadtxt(f'int_MC_N{N_MC}_{thermo}.txt')
temp = TNIQ1[:, 0].reshape(-1,nrepl)

intg1 = TNIQ1[:, 2].reshape(-1,nrepl)
intg1_qt = np.quantile(intg1, [0.25, 0.5, 0.75], axis=1).T

TNIQ2 = np.loadtxt(f'int_GMMT{suffix}_N{N_GMMT}_{thermo}.txt')

intg2 = TNIQ2[:, 2].reshape(-1,nrepl)
intg2_qt = np.quantile(intg2, [0.25, 0.5, 0.75], axis=1).T

plt.plot(temp[:,0],intg1_qt[:, 1], 'o-', color='r', ms=6, label=rf'MC N={N_MC}')
plt.fill_between(temp[:, 0], intg1_qt[:, 0], intg1_qt[:, 2],
                 color=lighten_color('r', 0.5), alpha=0.5)

plt.plot(temp[:,0],intg2_qt[:, 1], 'o-', color='g', ms=6, label=rf'GMMT N={N_GMMT}')
plt.fill_between(temp[:, 0], intg2_qt[:, 0], intg2_qt[:, 2],
                 color=lighten_color('g', 0.5), zorder=-100000, alpha=0.5)

if thermo=='I0':
    intg1_kb=np.loadtxt(f'int_mc_kb_N100000.txt')
    intg2_kb=np.loadtxt(f'int_gmmt_kb_N10000.txt')

    plt.plot(temp[:,0], intg1_kb, 'bo-', ms=6, label=f'MC(KB) N=100000')
    plt.plot(temp[:,0], intg2_kb, 'ko-', ms=6, label=f'GMMT(KB) N=10000')

plt.xlabel('Temp.')
plt.ylabel(f'{thermo}')
plt.legend(loc='upper left')
plt.savefig(f'intg_mc_gmmt{suffix}_{thermo}.png')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'intg_mc_gmmt{suffix}_{thermo}_log.png')

