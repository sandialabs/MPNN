#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from klpc.utils.plotting import myrc, lighten_color

myrc()

#[100 200 400 800 1600 3200 6400 12800 25600 51200 102400]
nc=11
NN_all = [100*2**i for i in range(nc)]
thermo='I0'
nrepl = 50

#nt=31
#TT_all = [290+37*i for i in range(nt)] # Hardwired
nt=12
TT_all = [300+100*i for i in range(nt)] # Hardwired

colors = ['r', 'g']


TNIQ_all=np.empty((nrepl*nt, 4, nc, 2))
for i, NN in enumerate(NN_all):
    TNIQ_all[:, :, i, 0] = np.loadtxt(f'int_MC_N{NN}_{thermo}.txt')
    TNIQ_all[:, :, i, 1] = np.loadtxt(f'int_GMMT_N{NN}_{thermo}.txt')

for j, TT in enumerate(TT_all):
    plt.figure(figsize=(12,10))
    for m, meth in enumerate(['MC', 'GMMT']):
        intg = TNIQ_all[j*nrepl:(j+1)*nrepl, 2, :, m].T
        #print(intg.shape)
        #intg -= intg[-1,:]
        #intg = np.abs(intg)
        intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T
        #print(intg_qt.shape)

        plt.plot(NN_all,intg_qt[:, 1], 'o-', color=colors[m], ms=6, label=meth)
        plt.fill_between(NN_all, intg_qt[:, 0], intg_qt[:, 2],
                         color=lighten_color(colors[m], 0.5), alpha=0.5)
        plt.title(f'T={int(TT)}')


        plt.xlabel('Number of samples, N')
        plt.ylabel(thermo)
        plt.legend(loc='lower right')
        plt.xscale('log')
        plt.savefig(f'intg_conv_{thermo}_{int(TT)}.png')
        #plt.yscale('log')
        #plt.legend(loc='lower right')
        #plt.savefig(f'intg_conv_{int(TT)}_log.png')
