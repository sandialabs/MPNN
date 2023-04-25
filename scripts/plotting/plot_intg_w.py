#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import myrc, lighten_color

myrc()

thermo='I2'
N_MC=1000000
N_GMMT=100000

colors = ['r', 'm', 'b', 'g']

nrepl = 10

plt.figure(figsize=(12,10))

i=0
for ww in [1,4,16]:

    TNIQ = np.loadtxt(f'int_GMMT_w{ww}_N{N_GMMT}_{thermo}.txt')
    temp = TNIQ[:, 0].reshape(-1,nrepl)

    intg = TNIQ[:, 2].reshape(-1,nrepl)
    intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T

    plt.plot(temp[:,0],intg_qt[:, 1], 'o-', color=colors[i], ms=6, label=rf'GMMT N={N_GMMT}, w={ww}')
    plt.fill_between(temp[:, 0], intg_qt[:, 0], intg_qt[:, 2],
                 color=lighten_color(colors[i], 0.5), alpha=0.5)

    i+=1

TNIQ2 = np.loadtxt(f'int_MC_N{N_MC}_{thermo}.txt')
intg2 = TNIQ2[:, 2].reshape(-1,nrepl)
intg2_qt = np.quantile(intg2, [0.25, 0.5, 0.75], axis=1).T

plt.plot(temp[:,0],intg2_qt[:, 1], 'o-', color=colors[i], ms=6, label=rf'MC N={N_MC}')
plt.fill_between(temp[:, 0], intg2_qt[:, 0], intg2_qt[:, 2],
             color=lighten_color(colors[i], 0.5), alpha=0.5)

if thermo=='I0':
    intg2_kb=np.loadtxt(f'int_gmmt_kb_N10000.txt')
    plt.plot(temp[:,0], intg2_kb, 'ko-', ms=6, label=f'GMMT(KB) N=10000')

plt.xlabel('Temp.')
plt.ylabel(f'{thermo}')
plt.legend(loc='upper left')
plt.savefig(f'intg_gmmt_N{N_GMMT}_{thermo}.png')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'intg_gmmt_N{N_GMMT}_{thermo}_log.png')

