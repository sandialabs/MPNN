#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import myrc, lighten_color

myrc()


# method = 'MC'
# suffix=''
# thermo='I2'
# NN_all = [1000, 10000, 100000, 1000000]
# kbpow=5

method = 'GMMT'
thermo='I2'
NN_all = [100, 1000, 10000, 100000]

colors = ['r', 'm', 'b', 'g']

plt.figure(figsize=(12,10))

for i, NN in enumerate(NN_all):
    TNIQ = np.loadtxt(f'int_{method}_N{NN}_{thermo}.txt')
    nrepl = 50

    temp = TNIQ[:, 0].reshape(-1,nrepl)

    intg = TNIQ[:, 2].reshape(-1,nrepl)
    intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T


    plt.plot(temp[:,0],intg_qt[:, 1], 'o-', color=colors[i], ms=6, label=rf'{method} N=10$^{i+2}$')
    plt.fill_between(temp[:, 0], intg_qt[:, 0], intg_qt[:, 2],
                     color=lighten_color(colors[i], 0.5), alpha=0.5)
    i+=1

plt.xlabel('Temp.')
plt.ylabel(thermo)
plt.legend(loc='upper left')
plt.savefig(f'intg_{method}_{thermo}.png')
plt.yscale('log')
plt.legend(loc='lower right')
plt.savefig(f'intg_{method}_{thermo}_log.png')

