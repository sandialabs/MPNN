#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import myrc, lighten_color

myrc()

NN=100000
method, ww, ii = 'GMMT', '_w16', '2'




thermo, icol = 'I'+ii, 2+int(ii)


nrepl = 10

plt.figure(figsize=(12,10))

i=0

TNIQ = np.loadtxt(f'int_{method}{ww}_N{NN}_{thermo}.txt')
temp = TNIQ[:, 0].reshape(-1,nrepl)

intg = TNIQ[:, 2].reshape(-1,nrepl)
intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T

plt.plot(temp[:,0],intg_qt[:, 1], 'o-', color='r', ms=6, label='Single eval.')
plt.fill_between(temp[:, 0], intg_qt[:, 0], intg_qt[:, 2], color=lighten_color('r', 0.5), alpha=0.5)


TNIQ2 = np.loadtxt(f'int_{method}{ww}_N{NN}_multi.txt')
intg2 = TNIQ2[:, icol].reshape(-1,nrepl)
intg2_qt = np.quantile(intg2, [0.25, 0.5, 0.75], axis=1).T

plt.plot(temp[:,0],intg2_qt[:, 1], 'o-', color='g', ms=6, label='Multi eval.')
plt.fill_between(temp[:, 0], intg2_qt[:, 0], intg2_qt[:, 2], color=lighten_color('g', 0.5), alpha=0.5)


plt.title(rf'{method.upper()} N={NN}')
plt.xlabel('Temp.')
plt.ylabel(thermo)
plt.legend(loc='upper left')
plt.savefig(f'intg_{method}{ww}_sm_N{NN}_{thermo}.png')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'intg_{method}{ww}_sm_N{NN}_{thermo}_log.png')

