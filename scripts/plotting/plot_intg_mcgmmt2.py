#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from klpc.utils.plotting import myrc, lighten_color

myrc()

NNs = [100, 1000, 10000, 100000]
fmts = [':', '-.', '--', '-']
thermo='I0'

nrepl = 50

plt.figure(figsize=(12,10))

for i, NN in enumerate(NNs):
    fmt = fmts[i]
    TNIQ1 = np.loadtxt(f'int_MC_N{NN}_{thermo}.txt')
    temp = TNIQ1[:, 0].reshape(-1,nrepl)

    intg1 = TNIQ1[:, 2].reshape(-1,nrepl)
    intg1_qt = np.quantile(intg1, [0.25, 0.5, 0.75], axis=1).T

    TNIQ2 = np.loadtxt(f'int_GMMT_N{NN}_{thermo}.txt')

    intg2 = TNIQ2[:, 2].reshape(-1,nrepl)
    intg2_qt = np.quantile(intg2, [0.25, 0.5, 0.75], axis=1).T

    plt.plot(temp[:,0],intg1_qt[:, 1], 'o'+fmt, color='r', ms=6, label=rf'MC      N=10$^{i+2}$')
    # plt.fill_between(temp[:, 0], intg1_qt[:, 0], intg1_qt[:, 2],
    #                  color=lighten_color('r', 0.5), alpha=0.5)

    plt.plot(temp[:,0],intg2_qt[:, 1], 'o'+fmt, color='g', ms=6, label=rf'GMMT N=10$^{i+2}$')
    # plt.fill_between(temp[:, 0], intg2_qt[:, 0], intg2_qt[:, 2],
    #                  color=lighten_color('g', 0.5), zorder=-100000, alpha=0.5)


plt.xlabel('Temp.')
plt.ylabel(f'{thermo}')
plt.legend(loc='upper left')
plt.savefig(f'intg_mc_gmmt_{thermo}.png')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'intg_mc_gmmt_{thermo}_log.png')

