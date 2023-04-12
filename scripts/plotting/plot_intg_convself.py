#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from klpc.utils.plotting import myrc, lighten_color

myrc()

#[100 200 400 800 1600 3200 6400 12800 25600 51200 102400]
nc=11
NN_all = [100*2**i for i in range(nc)]
method = 'GMMT'
suffix='_w16'
thermo='I0'
nrepl = 100 #10

# nt=31
# TT_all = [290+37*i for i in range(nt)] # Hardwired
nt=3
TT_all = [300+250*i for i in range(nt)] # Hardwired

colors = ['r', 'm', 'b', 'g', 'orange', 'y', 'grey']


TNIQ_all=np.empty((nrepl*nt, 4, nc))
for i, NN in enumerate(NN_all):
    TNIQ_all[:, :, i] = np.loadtxt(f'int_{method}{suffix}_N{NN}_{thermo}.txt')

plt.figure(figsize=(12,10))
i=0
for j, TT in enumerate(TT_all): #TT_all[::6]
    intg = TNIQ_all[j*nrepl:(j+1)*nrepl, 2, :].T
    #print(intg.shape)
    intg = np.abs(intg-intg[-1,:])/np.abs(intg[-1,:])

    intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T
    intg_stat = intg_qt[:,1]
    #intg_stat = np.mean(intg, axis=1)
    #print(intg_qt.shape)



    plt.plot(NN_all,intg_stat, 'o-', color=colors[i], ms=6, label=f'T={int(TT)}')
    #plt.fill_between(NN_all, intg_qt[:, 0], intg_qt[:, 2],
    #                 color=lighten_color(colors[i%4], 0.5), alpha=0.5)
    #plt.title(f'T={int(TT)}')
    i+=1


plt.xlabel(r'Number of samples, $N$')
plt.ylabel(r'Relative error,  $\frac{|I_0(N)-I_0(10^5)|}{I_0(10^5)}$')
plt.legend(loc='upper right')
plt.xscale('log')
plt.savefig(f'intg_conv.png')
# plt.yscale('log')
# plt.savefig(f'intg_conv_log.png')
plt.clf()

# Now plot self convergence
plt.figure(figsize=(12,10))
i=0
for j, TT in enumerate(TT_all): #TT_all[::6]
    intg = TNIQ_all[j*nrepl:(j+1)*nrepl, 2, :].T
    #print(intg.shape)
    #intg = np.abs(intg-intg[-1,:])/np.abs(intg[-1,:])

    intg_qt = np.quantile(intg, [0.25, 0.5, 0.75], axis=1).T
    intg_stat = intg_qt[:,1]
    #intg_stat = np.mean(intg, axis=1)
    #print(intg_qt.shape)



    plt.plot(NN_all[1:],np.abs(intg_stat[1:]-intg_stat[:-1]), 'o-', color=colors[i], ms=6, label=f'T={int(TT)}')
    i+=1


plt.xlabel(r'Number of samples, $N$')
plt.ylabel(r'Self-convergence,  $|I_0(N)-I_0(N/2)|$')
plt.legend(loc='upper right')
plt.xscale('log')
plt.yscale('log')
plt.savefig(f'intg_conv_self.png')


